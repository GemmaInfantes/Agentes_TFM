from langgraph.graph import StateGraph, START, END
from src.state import DocState
# Importar las funciones de cada agente
from src.agent_loader import run_loader
from src.agent_metadata import run_metadata
from src.agent_summarizer import run_summarizer
from src.agent_keywords import run_keywords
from src.agent_topics import run_topics
from src.agent_structure import run_structure
from src.agent_insights import run_insights
from src.vectorizer_agent import run_vectorizer
from src.indexer_agent import run_indexer

# (Opcional) Agente de depuración:
from src.agent_loader import json  # para usar json si hiciera falta
def run_debug(state: DocState) -> DocState:
    """
    Agente de depuración: imprime el estado (en JSON) en la consola, sin modificarlo.
    """
    state_json = json.dumps(state, ensure_ascii=False, indent=2)
    print("\n>>> [DebugAgent] Estado actual:\n", state_json, "\n>>> Fin del estado.\n")
    return state

def build_graph():
    # Crear el grafo de estado basado en nuestra estructura DocState
    builder = StateGraph(DocState)

    # Añadir nodos (agentes) al grafo:
    builder.add_node("LoaderAgent",    run_loader)
    builder.add_node("MetadataAgent",  run_metadata)
    # Nodos en paralelo (posteriores a MetadataAgent):
    builder.add_node("SummarizerAgent", run_summarizer)
    builder.add_node("KeywordAgent",    run_keywords)
    builder.add_node("TopicModelAgent", run_topics)
    builder.add_node("StructureAgent",  run_structure)
    builder.add_node("InsightAgent",    run_insights)
    # Nodo de debug (sin alterar estado, opcional)
    builder.add_node("DebugAgent",      run_debug)
    # Nodos finales
    builder.add_node("VectorizerAgent", run_vectorizer)
    builder.add_node("IndexerAgent",    run_indexer)

    # Definir las aristas (flujo entre agentes):
    builder.add_edge(START,           "LoaderAgent")      # inicio -> cargador
    builder.add_edge("LoaderAgent",   "MetadataAgent")    # cargador -> metadata

    # De MetadataAgent a cada agente de enriquecimiento (ejecución en paralelo lógica)
    builder.add_edge("MetadataAgent", "SummarizerAgent")
    builder.add_edge("MetadataAgent", "KeywordAgent")
    builder.add_edge("MetadataAgent", "TopicModelAgent")
    builder.add_edge("MetadataAgent", "StructureAgent")
    builder.add_edge("MetadataAgent", "InsightAgent")

    # Sincronización a través de DebugAgent:
    builder.add_edge("SummarizerAgent", "DebugAgent")
    builder.add_edge("KeywordAgent",    "DebugAgent")
    builder.add_edge("TopicModelAgent", "DebugAgent")
    builder.add_edge("StructureAgent",  "DebugAgent")
    builder.add_edge("InsightAgent",    "DebugAgent")

    # Continuación del flujo tras DebugAgent:
    builder.add_edge("DebugAgent",      "VectorizerAgent")
    builder.add_edge("VectorizerAgent", "IndexerAgent")
    builder.add_edge("IndexerAgent",    END)  # Fin del flujo

    # Compilar el grafo a un pipeline ejecutable
    pipeline = builder.compile()
    return pipeline
