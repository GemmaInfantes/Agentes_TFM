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
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        embeddings: lista de vectores (List[List[float]])
        metadata:  lista de diccionarios JSON (List[Dict[str, Any]])
        """
        logging.info(f"[IndexerAgent] Iniciando indexación con {len(embeddings)} embeddings y {len(metadata)} metadatos")
        
        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError("Se esperaba una lista no vacía de embeddings.")
        if not isinstance(metadata, list) or not metadata:
            raise ValueError("Se esperaba una lista no vacía de metadatos.")
        if len(embeddings) != len(metadata):
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

        # Validar que cada metadato es un dict plano
        metadata_limpios = []
        for idx, meta in enumerate(metadata):
            if not isinstance(meta, dict):
                raise ValueError(f"El metadato en la posición {idx} no es un dict.")
            # Eliminar claves anidadas no deseadas
            meta_plano = dict(meta)  # copia
            if "metadata" in meta_plano:
                inner = meta_plano.pop("metadata")
                if isinstance(inner, dict):
                    meta_plano.update(inner)
            if "metadatos" in meta_plano:
                inner = meta_plano.pop("metadatos")
                if isinstance(inner, dict):
                    meta_plano.update(inner)
            metadata_limpios.append(meta_plano)

        # LOGS DE DEPURACIÓN ANTES DE INSERTAR
        logging.info(f"[IndexerAgent] Embeddings a insertar: {len(embeddings)}")
        logging.info(f"[IndexerAgent] Metadata a insertar: {len(metadata_limpios)}")
        if metadata_limpios:
            logging.info(f"[IndexerAgent] Ejemplo de metadata: {metadata_limpios[0]}")
        else:
            logging.info(f"[IndexerAgent] Metadata está VACÍO")

        # Preparar los datos para insertar: 
        # Milvus espera una lista de columnas, en el mismo orden en que definimos los FieldSchema
        # (omitimos 'id' porque es auto_id):
        #   - columna embeddings (FLOAT_VECTOR)
        #   - columna metadatos  (JSON)
        insert_data = [
            embeddings,   # FIELD 1: FLOAT_VECTOR
            metadata_limpios     # FIELD 2: JSON (dict plano)
        ]
        logging.info(f"[IndexerAgent] Ejemplo de metadato a insertar: {metadata_limpios[0] if metadata_limpios else 'VACÍO'}")
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
    Toma state['embeddings'] y state['metadata'] y los inserta en Milvus.
    Por defecto, state['metadata'] es la lista de dicts que generaron los agentes anteriores.
    """
    embeddings = state.get("embeddings", [])
    metadata  = state.get("metadatos", [])

    if not embeddings or not metadata:
        raise ValueError("Faltan embeddings o metadatos en el estado para indexar.")

    result = indexer.run(embeddings, metadata)
    state["index_result"] = result
    return state