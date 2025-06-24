import os
import streamlit as st
from src.state import DocState
from src.graph_builder import build_graph
from typing import Dict, Any
import numpy as np
import json

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Documentos con Agentes",
    page_icon="📄",
    layout="wide"
)

# Título y descripción
st.title("🤖 Sistema de Análisis de Documentos con Agentes Inteligentes")
st.markdown("""
Este sistema analiza documentos utilizando múltiples agentes especializados para extraer información relevante.
""")

# Sidebar con configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("---")
    
    # Selector de etapas a mostrar en los resultados
    st.subheader("Selecciona las etapas a mostrar:")
    show_metadata = st.checkbox("Metadata", value=True)
    show_summary = st.checkbox("Resumen", value=True)
    show_keywords = st.checkbox("Palabras Clave", value=True)
    show_topics = st.checkbox("Temas", value=True)
    show_structure = st.checkbox("Estructura", value=True)
    show_insights = st.checkbox("Insights", value=True)

# Área principal
uploaded_file = st.file_uploader(
    "Selecciona un documento (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    # Crear directorio temporal si no existe
    if not os.path.exists("uploaded_docs"):
        os.makedirs("uploaded_docs")
    
    # Guardar el archivo
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.success(f"Archivo guardado: {uploaded_file.name}")
    
    # Botón para iniciar el procesamiento
    if st.button("🚀 Iniciar Análisis"):
        # Inicializar el estado
        state: DocState = {"file_path": file_path}
        
        # Contenedor para la barra de progreso
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Construir y ejecutar el grafo
                pipeline = build_graph()
                status_text.text("Procesando documento...")
                state = pipeline.invoke(state)
                progress_bar.progress(100)
                
                # Mostrar resultados
                st.success("¡Análisis completado!")
                
                # Mostrar resultados en pestañas
                visible_tabs = []
                if show_metadata:
                    visible_tabs.append("📊 Metadata")
                if show_summary:
                    visible_tabs.append("📝 Resumen")
                if show_keywords:
                    visible_tabs.append("🔑 Palabras Clave")
                if show_topics:
                    visible_tabs.append("📚 Temas")
                if show_structure:
                    visible_tabs.append("🏗️ Estructura")
                if show_insights:
                    visible_tabs.append("💡 Insights")
                
                tabs = st.tabs(visible_tabs)
                
                tab_idx = 0
                
                if show_metadata:
                    with tabs[tab_idx]:
                        # Mostrar source_stats como resumen general
                        if "source_stats" in state:
                            st.subheader("Estadísticas de carga")
                            st.json(state["source_stats"])
                        # Mostrar metadatos enriquecidos de cada documento
                        if "documents" in state:
                            for doc in state["documents"]:
                                st.markdown(f"**{doc['title']}**")
                                if "metadata" in doc:
                                    meta = doc["metadata"]
                                    # Mostrar campos clave de forma bonita, solo si tienen valor y sin duplicados
                                    shown_keys = set()
                                    for key in ["source", "author", "title_docx", "created", "last_modified_by", "last_printed", "modified", "category", "comments", "subject"]:
                                        if key not in shown_keys:
                                            value = meta.get(key, None)
                                            if value not in (None, "", "None"):
                                                st.write(f"**{key}:** {value}")
                                                shown_keys.add(key)
                                st.markdown("---")
                    tab_idx += 1
                
                if show_summary:
                    with tabs[tab_idx]:
                        if "documents" in state:
                            for doc in state["documents"]:
                                if "metadata" in doc:
                                    st.write(f"**{doc['title']}**")
                                    meta = doc["metadata"]
                                    # Mostrar resumen abstractivo
                                    if "summary_abstract" in meta:
                                        st.subheader("Resumen (abstractivo)")
                                        st.markdown(meta["summary_abstract"])
                                    # Mostrar resumen extractivo
                                    if "summary_extractive" in meta:
                                        st.subheader("Resumen (extractivo)")
                                        st.markdown(meta["summary_extractive"])
                                    st.markdown("---")
                    tab_idx += 1
                
                if show_keywords:
                    with tabs[tab_idx]:
                        if "documents" in state:
                            for doc in state["documents"]:
                                if "metadata" in doc and "keywords" in doc["metadata"]:
                                    st.write(f"**{doc['title']}**")
                                    st.write(", ".join(doc["metadata"]["keywords"]))
                                    st.markdown("---")
                    tab_idx += 1
                
                if show_topics:
                    with tabs[tab_idx]:
                        if "documents" in state:
                            for doc in state["documents"]:
                                if "metadata" in doc and "topics" in doc["metadata"]:
                                    st.write(f"**{doc['title']}**")
                                    st.write("\n".join(f"- {topic}" for topic in doc["metadata"]["topics"]))
                                    st.markdown("---")
                    tab_idx += 1
                
                if show_structure:
                    with tabs[tab_idx]:
                        if "documents" in state:
                            for doc in state["documents"]:
                                if "metadata" in doc and "structure" in doc["metadata"]:
                                    st.write(f"**{doc['title']}**")
                                    st.json(doc["metadata"]["structure"])
                                    st.markdown("---")
                    tab_idx += 1
                
                if show_insights:
                    with tabs[tab_idx]:
                        if "documents" in state:
                            for doc in state["documents"]:
                                if "metadata" in doc and "insights" in doc["metadata"]:
                                    st.write(f"**{doc['title']}**")
                                    st.write("\n".join(f"- {insight}" for insight in doc["metadata"]["insights"]))
                                    st.markdown("---")
                    tab_idx += 1
                
            except Exception as e:
                st.error(f"Error durante el procesamiento: {str(e)}")
            finally:
                # Limpiar archivo temporal
                if os.path.exists(file_path):
                    os.remove(file_path)
else:
    st.info("Por favor, sube un documento para comenzar el análisis.")
