import json
import logging
from typing import Dict, Any, List
from configs.openai_config import openai_llm
from src.state import DocState
from langchain.schema import SystemMessage, HumanMessage

# 2.6) StructureAgent
def extract_structure(inputs: dict) -> dict:
    """
    Extrae la estructura jerárquica de cada documento usando el modelo de lenguaje.
    """
    docs = inputs.get('documents', [])
    enriched = []
    for doc in docs:
        title = doc.get('title')
        text = doc.get('text', '')

        prompt = (
            f"Eres un asistente experto en analizar la estructura de documentos legales. "
            f"Analiza el siguiente texto y extrae su estructura jerárquica. "
            f"Identifica las secciones principales y sus subsecciones. "
            f"Devuelve un JSON con la siguiente estructura exacta:\n"
            f"{{\n"
            f"  \"structure\": [\n"
            f"    {{\n"
            f"      \"section_title\": \"NOMBRE DE LA SECCIÓN\",\n"
            f"      \"subsections\": [\"SUBSECCIÓN 1\", \"SUBSECCIÓN 2\", ...]\n"
            f"    }}\n"
            f"  ]\n"
            f"}}\n\n"
            f"Texto a analizar:\n{text[:5000]}"
        )
        
        try:
            messages = [
                SystemMessage(content="Eres un asistente experto en analizar la estructura de documentos legales."),
                HumanMessage(content=prompt)
            ]
            out = openai_llm.invoke(messages)
            logging.info(f"[StructureAgent] Respuesta del modelo para {title}: {out.content[:200]}...")
            
            # Limpiar la respuesta de posibles prefijos de markdown
            content = out.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            res = json.loads(content)
            structure = res.get('structure', [])
            logging.info(f"[StructureAgent] Estructura extraída para {title}: {json.dumps(structure, indent=2)}")
        except json.JSONDecodeError as e:
            logging.warning(f"[StructureAgent] Error parseando estructura de «{title}»: {str(e)}")
            logging.warning(f"[StructureAgent] Contenido recibido: {out.content}")
            structure = []
        except Exception as e:
            logging.error(f"[StructureAgent] Error al extraer estructura: {str(e)}")
            structure = []

        meta = doc.get('metadata', {})
        meta['structure'] = structure
        enriched.append({
            'title': title,
            'text': text,
            'metadata': meta
        })

    inputs['documents'] = enriched
    logging.info(f"[StructureAgent] Añadida estructura a {len(enriched)} documentos")
    return inputs


def run_structure(state: DocState) -> DocState:
    """
    Enriquece cada documento con su estructura jerárquica de secciones.
    """
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }

    result = extract_structure(payload)

    # Actualizar la estructura en cada documento
    for idx, doc_enriquecido in enumerate(result["documents"]):
        structure = doc_enriquecido.get("metadata", {}).get("structure", [])
        if "metadatos" not in state:
            state["metadatos"] = []
        if len(state["metadatos"]) <= idx:
            state["metadatos"].append({})
        state["metadatos"][idx]["structure"] = structure

    return state
