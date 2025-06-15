import os
import logging
from typing import Dict, Any, Optional
from langdetect import detect as langdetect_detect
from src.state import DocState

def detect(text: str) -> dict:
    """
    Detecta el idioma principal de un texto usando langdetect.
    Retorna un dict con 'lang' y 'prob' (idioma y probabilidad).
    """
    try:
        lang = langdetect_detect(text)
        return {"lang": lang, "prob": 1.0}  # langdetect no proporciona probabilidades
    except:
        return {"lang": "unknown", "prob": 0.0}

def extract_metadata(inputs: dict) -> dict:
    """
    Toma inputs={'documents': [...], 'source_stats': {...}} y en cada documento
    añade 'language' y 'token_count' en inputs['documents'][i]['metadata'].
    """
    docs_list = inputs.get('documents', [])
    enriched = []
    for doc in docs_list:
        text = doc.get('text', '')
        base_meta = doc.get('metadata', {})
        language = detect(text[:2000]) if text else {"lang": "unknown", "prob": 0.0}
        token_count = len(text.split())

        new_meta = {**base_meta, 'language': language, 'token_count': token_count}
        enriched.append({
            'title': doc.get('title'),
            'text': text,
            'metadata': new_meta
        })

    inputs['documents'] = enriched
    logging.info(f"[MetadataAgent] Enriquecidos {len(enriched)} documentos con idioma y token_count")
    return inputs

def run_metadata(state: DocState) -> DocState:
    """
    Agente que enriquece los documentos con metadatos adicionales:
    - Detección de idioma
    - Conteo de tokens
    """
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }
    result = extract_metadata(payload)
    state["documents"] = result["documents"]
    return state