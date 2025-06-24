import logging
from typing import Any, List, Dict
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from src.state import DocState

class IndexerAgent:
    """
    Agente para indexar embeddings + metadatos JSON en Milvus.
    """
    def __init__(self, 
                 collection_name: str = "documentos_legales_v2", 
                 host: str = "localhost", 
                 port: str = "19530"):
        self.collection_name = collection_name
        connections.connect(alias="default", host=host, port=port)
        self.collection = None
        logging.info(f"[IndexerAgent] Inicializado con colección: {self.collection_name}")

    def _create_collection(self, dim: int):
        """
        Crea una colección con 3 campos: 
          - id       (INT64, primary, auto_id)
          - embedding (FLOAT_VECTOR, dimension=dim)
          - metadata  (JSON)
        """
        fields = [
            FieldSchema(
                name="id", 
                dtype=DataType.INT64, 
                is_primary=True, 
                auto_id=True
            ),
            FieldSchema(
                name="embedding", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=dim
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON
            )
        ]
        schema = CollectionSchema(fields, description="Colección de embeddings R50 + metadatos JSON")
        self.collection = Collection(name=self.collection_name, schema=schema)
        logging.info(f"[IndexerAgent] Colección '{self.collection_name}' creada con dimensión {dim}.")

    def _get_or_create_collection(self, dim: int):
        """
        Si la colección existe, comprueba que su dimensión coincide. Si no existe, la crea.
        """
        if self.collection is None:
            if self.collection_name in utility.list_collections():
                self.collection = Collection(self.collection_name)
                existing_dim = self.collection.schema.fields[1].params["dim"]
                if existing_dim != dim:
                    raise ValueError(
                        f"Dimensión existente ({existing_dim}) no coincide con la solicitada ({dim})."
                    )
                logging.info(f"[IndexerAgent] Conectado a colección existente '{self.collection_name}' (dim={dim}).")
            else:
                self._create_collection(dim)

    def run(
        self,
        embeddings: List[List[float]],
        metadatos: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        embeddings: lista de vectores (List[List[float]])
        metadatos:  lista de diccionarios JSON (List[Dict[str, Any]])
        """
        logging.info(f"[IndexerAgent] Iniciando indexación con {len(embeddings)} embeddings y {len(metadatos)} metadatos")
        
        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError("Se esperaba una lista no vacía de embeddings.")
        if not isinstance(metadatos, list) or not metadatos:
            raise ValueError("Se esperaba una lista no vacía de metadatos.")
        if len(embeddings) != len(metadatos):
            raise ValueError("La longitud de embeddings y metadatos debe coincidir.")

        dim = len(embeddings[0])
        logging.info(f"[IndexerAgent] Dimensión de embeddings: {dim}")
        for idx, vec in enumerate(embeddings):
            if len(vec) != dim:
                raise ValueError(
                    f"Dimensión del embedding en índice {idx} ({len(vec)}) no coincide con {dim}."
                )

        # Conectarse o crear la colección según sea necesario
        self._get_or_create_collection(dim)

        # Preparar los datos para insertar: 
        # Milvus espera una lista de columnas, en el mismo orden en que definimos los FieldSchema
        # (omitimos 'id' porque es auto_id):
        #   - columna embeddings (FLOAT_VECTOR)
        #   - columna metadatos  (JSON)
        insert_data = [
            embeddings,   # FIELD 1: FLOAT_VECTOR
            metadatos     # FIELD 2: JSON
        ]
        logging.info(f"[IndexerAgent] Preparando inserción de datos...")
        
        try:
            insert_result = self.collection.insert(insert_data)
            primary_keys = insert_result.primary_keys
            insert_count = len(primary_keys)
            logging.info(f"[IndexerAgent] Datos insertados exitosamente. Primary keys: {primary_keys}")
        except Exception as e:
            logging.error(f"[IndexerAgent] Error al insertar datos: {str(e)}")
            raise e

        # Crear índice para permitir búsquedas
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        logging.info(f"[IndexerAgent] Creando índice...")
        try:
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logging.info(f"[IndexerAgent] Índice creado para embeddings en '{self.collection_name}'.")
        except Exception as e:
            logging.error(f"[IndexerAgent] Error al crear índice: {str(e)}")
            raise e

        # Cargar la colección en memoria para permitir búsquedas
        logging.info(f"[IndexerAgent] Cargando colección en memoria...")
        try:
            self.collection.load()
            logging.info(f"[IndexerAgent] Colección '{self.collection_name}' cargada en memoria.")
        except Exception as e:
            logging.error(f"[IndexerAgent] Error al cargar colección: {str(e)}")
            raise e

        logging.info(f"[IndexerAgent] Insertados {insert_count} vectores+metadatos en '{self.collection_name}'.")
        return {"insert_count": insert_count, "primary_keys": primary_keys}


# Instancia global para no reconectar en cada llamada
indexer = IndexerAgent()

def run_indexer(state: DocState) -> DocState:
    """
    Toma state['embeddings'] y state['metadatos'] y los inserta en Milvus.
    Por defecto, state['metadatos'] es la lista de dicts que generaron los agentes anteriores.
    """
    embeddings = state.get("embeddings", [])
    metadatos  = state.get("metadatos", [])

    if not embeddings or not metadatos:
        raise ValueError("Faltan embeddings o metadatos en el estado para indexar.")

    result = indexer.run(embeddings, metadatos)
    state["index_result"] = result
    return state