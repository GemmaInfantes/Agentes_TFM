import os
import logging
from typing import Dict, Any, Optional
from langdetect import detect as langdetect_detect
from src.state import DocState
import re
import hashlib
import dateparser.search

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

def extract_dates(text: str):
    try:
        results = dateparser.search.search_dates(text, languages=["es", "en"])
        if results:
            return list(set([str(date[1].date()) for date in results if date[1]]))
        return []
    except Exception as e:
        logging.warning(f"Error extrayendo fechas: {e}")
        return []

def extract_author(text: str, meta: dict):
    # Busca en metadatos primero
    if 'author' in meta and meta['author']:
        return meta['author']
    # Busca patrones comunes en el texto
    match = re.search(r'(Autor|Author|Por|By)\s*[:\-]?\s*([\w\s,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return None

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def extract_metadata(inputs: dict, known_hashes=None) -> dict:
    """
    Toma inputs={'documents': [...], 'source_stats': {...}} y en cada documento
    añade 'language', 'token_count', 'dates', 'author', 'hash' y 'is_duplicate' en inputs['documents'][i]['metadata'].
    """
    docs_list = inputs.get('documents', [])
    enriched = []
    if known_hashes is None:
        known_hashes = set()
    for doc in docs_list:
        text = doc.get('text', '')
        base_meta = doc.get('metadata', {})
        language = detect(text[:2000]) if text else {"lang": "unknown", "prob": 0.0}
        token_count = len(text.split())
        # NUEVO: fechas y autor
        dates = extract_dates(text)
        author = extract_author(text, base_meta)
        # NUEVO: hash y duplicado
        doc_hash = compute_hash(text)
        is_duplicate = doc_hash in known_hashes
        known_hashes.add(doc_hash)
        new_meta = {**base_meta, 'language': language, 'token_count': token_count, 'dates': dates, 'author': author, 'hash': doc_hash, 'is_duplicate': is_duplicate}
        enriched.append({
            'title': doc.get('title'),
            'text': text,
            'metadata': new_meta
        })

    inputs['documents'] = enriched
    logging.info(f"[MetadataAgent] Enriquecidos {len(enriched)} documentos con idioma, token_count, fechas, autor y hash")
    return inputs

def run_metadata(state: DocState) -> DocState:
    """
    Agente que enriquece los documentos con metadatos adicionales:
    - Detección de idioma
    - Conteo de tokens
    - Extracción de fechas
    - Extracción de autor
    - Cálculo de hash SHA256
    - Detección de duplicados
    """
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }
    # Mantén un set de hashes ya vistos en el state
    if "known_hashes" not in state:
        state["known_hashes"] = set()
    result = extract_metadata(payload, known_hashes=state["known_hashes"])
    # Actualizar los metadatos en state['metadatos']
    for idx, doc_enriquecido in enumerate(result["documents"]):
        meta = doc_enriquecido.get("metadata", {})
        for key in ["language", "token_count", "dates", "author", "hash", "is_duplicate"]:
            if "metadatos" not in state:
                state["metadatos"] = []
            while len(state["metadatos"]) <= idx:
                state["metadatos"].append({})
            state["metadatos"][idx][key] = meta.get(key)
    return state