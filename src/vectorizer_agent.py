import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.state import DocState
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
import json

# --------------------------------------------------------------------
# 3) VectorizerAgent e IndexerAgent (con wrappers)
# --------------------------------------------------------------------

class VectorizerAgent:
    """
    Agente para generar embeddings usando SentenceTransformer("all-MiniLM-L6-v2").
    Ahora extrae solo el campo 'text' de cada documento y conserva doc['metadata'] aparte.
    """
    def __init__(self):
        try:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logging.error(f"Error al cargar SentenceTransformer: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo de embeddings: {e}")

    def _embed_text(self, text: str) -> List[float]:
        # Devuelve un vector normalizado en forma de lista de floats
        return self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def run(self, docs: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        docs: lista de diccionarios, cada uno con al menos:
            { "text": <string>, "metadata": <dict con todos sus campos> }
        Retorna un dict con dos listas paralelas:
            {
              "embeddings": [[...], [...], ...],
              "metadatos":  [{...}, {...}, ...]
            }
        """
        if not isinstance(docs, list):
            logging.error(f"Se esperaba una lista de documentos, se recibió: {type(docs)}")
            raise TypeError(f"Se esperaba una lista de documentos, se recibió: {type(docs)}")
        if not docs:
            logging.warning("Lista de documentos vacía, no hay vectores que generar.")
            return {"embeddings": [], "metadatos": []}

        logging.info(f"[VectorizerAgent] Procesando {len(docs)} documentos")
        lista_embeddings: List[List[float]] = []
        lista_metadatos:  List[Dict[str, Any]] = []

        for idx, entry in enumerate(docs):
            logging.info(f"[VectorizerAgent] Procesando documento {idx}")
            if not isinstance(entry, dict):
                logging.error(f"[VectorizerAgent] Elemento no es dict (índice {idx}): {entry}")
                raise TypeError(f"Elemento {idx} de la lista no es un dict con 'text' y 'metadata'.")
            
            # Intentar obtener el texto del documento
            texto = ""
            if "text" in entry:
                texto = entry["text"]
            elif "content" in entry:
                texto = entry["content"]
            elif "summary" in entry:
                texto = entry["summary"]
            
            if not texto:
                logging.error(f"[VectorizerAgent] No se encontró texto para procesar en el documento {idx}")
                raise ValueError(f"Documento {idx} no tiene texto para procesar")
            
            metadato_completo = entry

            logging.info(f"[VectorizerAgent] Documento {idx} - Longitud del texto: {len(texto)}")
            logging.info(f"[VectorizerAgent] Documento {idx} - Campos disponibles: {list(entry.keys())}")

            try:
                vec = self._embed_text(texto)
                lista_embeddings.append(vec)
                lista_metadatos.append(metadato_completo)
                logging.info(f"[VectorizerAgent] Documento {idx} procesado exitosamente")
            except Exception as e:
                logging.error(f"[VectorizerAgent] Error al generar embedding para el documento {idx}: {e}")
                raise RuntimeError(f"Error al generar embedding para el documento {idx}: {e}")

        logging.info(f"[VectorizerAgent] Procesamiento completado. Generados {len(lista_embeddings)} vectores")
        return {"embeddings": lista_embeddings, "metadatos": lista_metadatos}


# Instancia global para no recargar el modelo en cada llamada
vectorizer = VectorizerAgent()

def run_vectorizer(state: DocState) -> DocState:
    """
    Lee state['documents'] (lista de dicts con el texto y metadatos), 
    genera embeddings y guarda en el estado:
      - 'embeddings': lista de vectores
      - 'metadatos': lista de dicts con todos los metadatos originales
    """
    # Obtener los documentos del estado
    docs = state.get("documents", [])
    
    logging.info(f"[VectorizerAgent] Estado recibido - Campos disponibles: {list(state.keys())}")
    logging.info(f"[VectorizerAgent] Estado recibido - Número de documentos: {len(docs)}")
    
    if not docs:
        logging.warning("[VectorizerAgent] No hay documentos en el estado para procesar")
        return state
    
    # Log del primer documento para verificar su estructura
    if docs:
        logging.info(f"[VectorizerAgent] Estructura del primer documento: {list(docs[0].keys())}")
        if "text" in docs[0]:
            logging.info(f"[VectorizerAgent] Longitud del texto del primer documento: {len(docs[0]['text'])}")
        elif "content" in docs[0]:
            logging.info(f"[VectorizerAgent] Longitud del contenido del primer documento: {len(docs[0]['content'])}")
        elif "summary" in docs[0]:
            logging.info(f"[VectorizerAgent] Longitud del resumen del primer documento: {len(docs[0]['summary'])}")
        
    resultado = vectorizer.run(docs)
    state["embeddings"] = resultado["embeddings"]
    state["metadatos"] = resultado["metadatos"]
    
    logging.info(f"[VectorizerAgent] Procesamiento completado:")
    logging.info(f"[VectorizerAgent] - Número de embeddings generados: {len(resultado['embeddings'])}")
    logging.info(f"[VectorizerAgent] - Número de metadatos procesados: {len(resultado['metadatos'])}")
    
    logging.info(f"Ejemplo de metadato a insertar: {json.dumps(resultado['metadatos'][0], indent=2, ensure_ascii=False)}")
    
    return state

connections.connect(alias="default", host="localhost", port="19530")

collection_name = "documentos_legales_v2"

if collection_name not in utility.list_collections():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, description="Colección de embeddings y metadatos")
    c = Collection(name=collection_name, schema=schema)
    print("Colección creada")
else:
    c = Collection(collection_name)
    print("Colección ya existe")

print("Número de entidades:", c.num_entities)

