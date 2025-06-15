import json
import logging
from typing import Dict, Any, List
from configs.openai_config import openai_llm
from src.state import DocState
from langchain.schema import SystemMessage, HumanMessage

# 2.5) TopicModelAgent
def extract_topics(inputs: dict) -> dict:
    """
    Extrae los temas principales de cada documento usando el modelo de lenguaje.
    """
    docs = inputs.get('documents', [])
    enriched = []
    for doc in docs:
        title = doc.get('title')
        text = doc.get('text', '')

        # Asegurarse de que el texto no esté vacío
        if not text.strip():
            logging.warning(f"[TopicAgent] Texto vacío para el documento {title}")
            topics = []
        else:
            prompt = (
                f"Eres un asistente experto en identificar temas clave de documentos. "
                f"Analiza el siguiente texto y extrae los 5 temas principales que trata. "
                f"Los temas deben ser específicos y relevantes para el contexto del documento. "
                f"Devuelve un JSON con la siguiente estructura exacta:\n"
                f"{{\n"
                f"  \"topics\": [\n"
                f"    \"TEMA 1\",\n"
                f"    \"TEMA 2\",\n"
                f"    \"TEMA 3\",\n"
                f"    \"TEMA 4\",\n"
                f"    \"TEMA 5\"\n"
                f"  ]\n"
                f"}}\n\n"
                f"Texto a analizar:\n{text[:5000]}"
            )
            
            try:
                messages = [
                    SystemMessage(content="Eres un asistente experto en identificar temas clave de documentos."),
                    HumanMessage(content=prompt)
                ]
                out = openai_llm.invoke(messages)
                logging.info(f"[TopicAgent] Respuesta del modelo para {title}: {out.content[:200]}...")
                
                # Limpiar la respuesta de posibles prefijos de markdown
                content = out.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                res = json.loads(content)
                topics = res.get('topics', [])
                logging.info(f"[TopicAgent] Temas extraídos para {title}: {json.dumps(topics, indent=2)}")
            except json.JSONDecodeError as e:
                logging.warning(f"[TopicAgent] Error parseando topics para '{title}': {str(e)}")
                logging.warning(f"[TopicAgent] Contenido recibido: {out.content}")
                topics = []
            except Exception as e:
                logging.error(f"[TopicAgent] Error al extraer topics: {str(e)}")
                topics = []

        meta = doc.get('metadata', {})
        meta['topics'] = topics
        enriched.append({
            'title': title,
            'text': text,
            'metadata': meta
        })

    inputs['documents'] = enriched
    logging.info(f"[TopicAgent] Añadidos topics a {len(enriched)} documentos")
    return inputs


def run_topics(state: DocState) -> DocState:
    """
    Enriquece cada documento con los temas principales extraídos del texto.
    """
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }

    result = extract_topics(payload)

    # Actualizar los topics en cada documento
    for idx, doc_enriquecido in enumerate(result["documents"]):
        if "metadata" not in state["documents"][idx]:
            state["documents"][idx]["metadata"] = {}
        
        state["documents"][idx]["metadata"]["topics"] = \
            doc_enriquecido.get("metadata", {}).get("topics", [])

    return state


