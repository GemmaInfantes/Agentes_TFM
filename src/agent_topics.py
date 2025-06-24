import json
import logging
from typing import Dict, Any, List
from configs.openai_config import openai_llm
from src.state import DocState
from langchain.schema import SystemMessage, HumanMessage
# NUEVO: BERTopic multilingüe
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logging.warning("BERTopic o sentence-transformers no están instalados, solo se usará el modelo LLM para topics.")

def extract_topics_bertopic(texts, language='multilingual'):
    if not BERTOPIC_AVAILABLE or len(texts) < 2:
        # No usar BERTopic si hay menos de 2 documentos
        return [[] for _ in texts], [[] for _ in texts]
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    topic_model = BERTopic(language=language, embedding_model=model, calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    topic_labels = [[topic_info.loc[topic, 'Name']] if topic in topic_info.index else [] for topic in topics]
    # Subtemas (si hay jerarquía)
    try:
        tree = topic_model.get_topic_tree()
        subtopics = [str(branch) for branch in tree.children]
    except Exception:
        subtopics = []
    return topic_labels, [subtopics for _ in texts]

def extract_topics(inputs: dict) -> dict:
    docs = inputs.get('documents', [])
    enriched = []
    texts = [doc.get('text', '') for doc in docs]
    # BERTopic solo si hay más de 1 documento
    topics_list, subtopics_list = extract_topics_bertopic(texts, language='multilingual') if len(docs) > 1 else ([[] for _ in docs], [[] for _ in docs])
    for i, doc in enumerate(docs):
        title = doc.get('title')
        text = doc.get('text', '')
        topics = topics_list[i] if topics_list else []
        subtopics = subtopics_list[i] if subtopics_list else []
        if not topics:
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
        meta['subtopics'] = subtopics
        enriched.append({
            'title': title,
            'text': text,
            'metadata': meta
        })
    inputs['documents'] = enriched
    logging.info(f"[TopicAgent] Añadidos topics a {len(enriched)} documentos")
    return inputs

def run_topics(state: DocState) -> DocState:
    payload = {
        "documents": state["documents"],
        "source_stats": state["source_stats"]
    }
    result = extract_topics(payload)
    for idx, doc_enriquecido in enumerate(result["documents"]):
        if "metadata" not in state["documents"][idx]:
            state["documents"][idx]["metadata"] = {}
        state["documents"][idx]["metadata"]["topics"] = doc_enriquecido.get("metadata", {}).get("topics", [])
        state["documents"][idx]["metadata"]["subtopics"] = doc_enriquecido.get("metadata", {}).get("subtopics", [])
    return state


