# 2.7) InsightAgent
import json
import logging
from typing import Dict, Any, List
from configs.openai_config import openai_llm
from src.state import DocState
from langchain.schema import SystemMessage, HumanMessage

def extract_insights(inputs: dict) -> dict:
    """
    Extrae insights y observaciones relevantes de cada documento usando el modelo de lenguaje.
    """
    docs = inputs.get('documents', [])
    enriched = []
    for doc in docs:
        title = doc.get('title')
        text = doc.get('text', '')

        prompt = (
            f"Eres un asistente experto en analizar documentos legales. "
            f"Analiza el siguiente texto y extrae 5 insights o observaciones relevantes. "
            f"Los insights deben ser puntos clave sobre obligaciones, derechos, plazos o condiciones importantes. "
            f"Devuelve un JSON con la siguiente estructura exacta:\n"
            f"{{\n"
            f"  \"insights\": [\n"
            f"    \"INSIGHT 1\",\n"
            f"    \"INSIGHT 2\",\n"
            f"    \"INSIGHT 3\",\n"
            f"    \"INSIGHT 4\",\n"
            f"    \"INSIGHT 5\"\n"
            f"  ]\n"
            f"}}\n\n"
            f"Texto a analizar:\n{text[:5000]}"
        )
        
        try:
            messages = [
                SystemMessage(content="Eres un asistente experto en analizar documentos legales."),
                HumanMessage(content=prompt)
            ]
            out = openai_llm.invoke(messages)
            logging.info(f"[InsightAgent] Respuesta del modelo para {title}: {out.content[:200]}...")
            
            # Limpiar la respuesta de posibles prefijos de markdown
            content = out.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            res = json.loads(content)
            insights = res.get('insights', [])
            logging.info(f"[InsightAgent] Insights extraídos para {title}: {json.dumps(insights, indent=2)}")
        except json.JSONDecodeError as e:
            logging.warning(f"[InsightAgent] Error parseando insights para '{title}': {str(e)}")
            logging.warning(f"[InsightAgent] Contenido recibido: {out.content}")
            insights = []
        except Exception as e:
            logging.error(f"[InsightAgent] Error al extraer insights: {str(e)}")
            insights = []

        meta = doc.get('metadata', {})
        meta['insights'] = insights
        enriched.append({
            'title': title,
            'text': text,
            'metadata': meta
        })

    inputs['documents'] = enriched
    logging.info(f"[InsightAgent] Añadidos insights a {len(enriched)} documentos")
    return inputs

def run_insights(state: DocState) -> DocState:
    """
    Enriquece cada documento con insights y observaciones relevantes.
    """
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }

    result = extract_insights(payload)

    # Actualizar los insights en cada documento
    for idx, doc_enriquecido in enumerate(result["documents"]):
        insights = doc_enriquecido.get("metadata", {}).get("insights", [])
        if "metadatos" not in state:
            state["metadatos"] = []
        if len(state["metadatos"]) <= idx:
            state["metadatos"].append({})
        state["metadatos"][idx]["insights"] = insights

    return state
