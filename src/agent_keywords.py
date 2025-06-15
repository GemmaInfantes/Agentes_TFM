import json
import logging
from typing import Dict, Any, List
from configs.openai_config import openai_llm
from src.state import DocState
from langchain.schema import SystemMessage, HumanMessage

def extract_keywords(inputs: dict) -> dict:
    """
    Extrae palabras clave de cada documento usando el modelo de lenguaje.
    """
    docs = inputs.get('documents', [])
    enriched = []
    for doc in docs:
        title = doc.get('title')
        text = doc.get('text', '')

        prompt = (
            f"Eres un asistente experto en identificar palabras clave de documentos legales. "
            f"Analiza el siguiente texto y extrae las 10 palabras clave más relevantes. "
            f"Las palabras clave deben ser términos específicos del ámbito legal y contractual. "
            f"Devuelve un JSON con la siguiente estructura exacta:\n"
            f"{{\n"
            f"  \"keywords\": [\n"
            f"    \"PALABRA CLAVE 1\",\n"
            f"    \"PALABRA CLAVE 2\",\n"
            f"    ...\n"
            f"  ]\n"
            f"}}\n\n"
            f"Texto a analizar:\n{text[:5000]}"
        )
        
        try:
            messages = [
                SystemMessage(content="Eres un asistente experto en identificar palabras clave de documentos legales."),
                HumanMessage(content=prompt)
            ]
            out = openai_llm.invoke(messages)
            logging.info(f"[KeywordAgent] Respuesta del modelo para {title}: {out.content[:200]}...")
            
            # Limpiar la respuesta de posibles prefijos de markdown
            content = out.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            res = json.loads(content)
            keywords = res.get('keywords', [])
            logging.info(f"[KeywordAgent] Palabras clave extraídas para {title}: {json.dumps(keywords, indent=2)}")
        except json.JSONDecodeError as e:
            logging.warning(f"[KeywordAgent] Error parseando keywords para '{title}': {str(e)}")
            logging.warning(f"[KeywordAgent] Contenido recibido: {out.content}")
            keywords = []
        except Exception as e:
            logging.error(f"[KeywordAgent] Error al extraer keywords: {str(e)}")
            keywords = []

        meta = doc.get('metadata', {})
        meta['keywords'] = keywords
        enriched.append({
            'title': title,
            'text': text,
            'metadata': meta
        })

    inputs['documents'] = enriched
    logging.info(f"[KeywordAgent] Añadidas keywords a {len(enriched)} documentos")
    return inputs
def run_keywords(state: DocState) -> DocState:
    """
    Enriquece cada documento con palabras clave extraídas del texto.
    """
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }

    result = extract_keywords(payload)

    # Actualizar las keywords en cada documento
    for idx, doc_enriquecido in enumerate(result["documents"]):
        keywords = doc_enriquecido.get("metadata", {}).get("keywords", [])
        if "metadatos" not in state:
            state["metadatos"] = []
        if len(state["metadatos"]) <= idx:
            state["metadatos"].append({})
        state["metadatos"][idx]["keywords"] = keywords

    return state
