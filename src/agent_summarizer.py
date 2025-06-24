import json
import logging
from typing import Dict, Any, List
from configs.openai_config import openai_llm
from src.state import DocState
from langchain.schema import SystemMessage, HumanMessage
# NUEVO: para resumen extractivo
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False
    logging.warning("sumy no está instalado, solo se usará resumen abstractivo.")

def extractive_summary(text, num_sentences=5, language='spanish'):
    if not SUMY_AVAILABLE:
        return None
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

def summarize(inputs: dict) -> dict:
    """
    Genera resúmenes, puntos clave y acciones recomendadas para cada documento.
    """
    docs = inputs.get('documents', [])
    summarized = []
    
    for doc in docs:
        title = doc.get('title')
        text = doc.get('text', '')
        # Resumen extractivo
        summary_extractive = extractive_summary(text, num_sentences=5, language='spanish') if text else None
        # Resumen abstractivo (LLM)
        prompt = (
            f"Eres un asistente experto en análisis documental. Resume el siguiente "
            f"documento de forma profesional en no más de 200 palabras. Si el texto "
            f"tiene múltiples secciones, incluye una oración clave por sección. No "
            f"repitas el texto, sintetízalo. "
            f"Genera un JSON con clave 'summary' (resumen breve) y 'key_points' (lista de bullets) "
            f"para el siguiente texto titulado '{title}':\n\n{text[:5000]}"
        )
        
        try:
            messages = [
                SystemMessage(content="Eres un asistente experto en análisis documental."),
                HumanMessage(content=prompt)
            ]
            out = openai_llm.invoke(messages)
            content = out.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            res = json.loads(content)
        except json.JSONDecodeError:
            res = {'summary': out.content.strip(), 'key_points': []}
        except Exception as e:
            logging.error(f"Error al generar resumen: {str(e)}")
            res = {
                'summary': "Error al generar resumen",
                'key_points': [],
                'error': str(e)
            }

        meta = doc.get('metadata', {})
        summary = res.get("summary")
        key_points = res.get("key_points", [])
        recommended_actions = res.get("recommended_actions", [])
        # NUEVO: guardar ambos resúmenes
        meta['summary_abstract'] = summary
        meta['summary_extractive'] = summary_extractive
        meta['key_points'] = key_points
        meta['recommended_actions'] = recommended_actions

        summarized.append({
            'title': title,
            'text': text,
            'metadata': meta
        })

    inputs['documents'] = summarized
    logging.info(f"[SummarizerAgent] Generados resúmenes para {len(summarized)} documentos")
    return inputs

def run_summarizer(state: DocState) -> DocState:
    """
    Enriquece cada documento con resúmenes, puntos clave y acciones recomendadas.
    """
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }

    result = summarize(payload)

    # Actualizar los metadatos de cada documento
    for idx, doc_enriquecido in enumerate(result["documents"]):
        meta_enriquecido = doc_enriquecido.get("metadata", {})
        summary_abstract = meta_enriquecido.get("summary_abstract")
        summary_extractive = meta_enriquecido.get("summary_extractive")
        key_points = meta_enriquecido.get("key_points", [])
        recommended_actions = meta_enriquecido.get("recommended_actions", [])
        if "metadatos" not in state:
            state["metadatos"] = []
        while len(state["metadatos"]) <= idx:
            state["metadatos"].append({})
        state["metadatos"][idx]["summary_abstract"] = summary_abstract
        state["metadatos"][idx]["summary_extractive"] = summary_extractive
        state["metadatos"][idx]["key_points"] = key_points
        state["metadatos"][idx]["recommended_actions"] = recommended_actions

    return state
