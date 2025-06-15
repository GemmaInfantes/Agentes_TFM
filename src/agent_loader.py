import time
import os
import json
import logging
import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_document(inputs: dict) -> dict:
    """
    Carga uno o varios documentos y devuelve {'documents': [...], 'source_stats': {...}}.
    """
    fp = inputs['file_path']
    if os.path.isdir(fp):
        paths = [os.path.join(fp, f) for f in sorted(os.listdir(fp))
                 if f.lower().endswith(('.pdf', '.txt', '.docx', '.doc'))]
    elif os.path.isfile(fp) and fp.lower().endswith(('.pdf', '.txt', '.docx', '.doc')):
        paths = [fp]
    else:
        raise ValueError(f"Ruta {fp} no es un archivo o carpeta válida.")

    documents = []
    total_pages = 0
    start = inputs.get('start_time', datetime.datetime.utcnow())
    
    for path in paths:
        try:
            ext = os.path.splitext(path)[1].lower()
            logger.info(f"Procesando archivo: {path} (extensión: {ext})")
            
            if ext == '.pdf':
                # Cargar PDF
                loader = PyPDFLoader(path)
                pages = loader.load()
                
                # Combinar todas las páginas en un solo documento
                combined_text = "\n".join(page.page_content for page in pages)
                
                # Extraer metadatos del PDF
                pdf_meta = {}
                if hasattr(loader, 'metadata'):
                    pdf_meta = loader.metadata
                
                # Crear un solo documento con el texto combinado
                documents.append({
                    'title': os.path.splitext(os.path.basename(path))[0],
                    'text': combined_text,
                    'metadata': {
                        'source': path,
                        'total_pages': len(pages),
                        'author': pdf_meta.get('author'),
                        'title_pdf': pdf_meta.get('title'),
                        'creation_date': pdf_meta.get('creation_date'),
                        'producer': pdf_meta.get('producer'),
                        'creator': pdf_meta.get('creator'),
                        'moddate': pdf_meta.get('moddate')
                    }
                })
                total_pages += len(pages)
                
            elif ext == '.txt':
                # Cargar archivo de texto
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(path, 'r', encoding='latin-1') as f:
                        content = f.read()
                
                documents.append({
                    'title': os.path.splitext(os.path.basename(path))[0],
                    'text': content,
                    'metadata': {
                        'source': path,
                        'total_pages': 1
                    }
                })
                total_pages += 1
                
            else:
                # Cargar otros tipos de documentos
                loader = UnstructuredWordDocumentLoader(path)
                pages = loader.load()
                text = "".join(p.page_content for p in pages)
                
                documents.append({
                    'title': os.path.splitext(os.path.basename(path))[0],
                    'text': text,
                    'metadata': {
                        'source': path,
                        'total_pages': len(pages)
                    }
                })
                total_pages += len(pages)
            
            logger.info(f"Archivo cargado exitosamente: {path}")
            
        except Exception as e:
            logger.error(f"Error al cargar el archivo {path}: {str(e)}")
            raise Exception(f"Error loading {path}: {str(e)}")

    stats = {
        'documents': len(documents),  # Ahora es el número real de documentos
        'total_pages': total_pages,
        'load_time_s': (datetime.datetime.utcnow() - start).total_seconds()
    }
    logger.info(f"[LoaderAgent] Cargados {stats['documents']} docs, {stats['total_pages']} páginas en {stats['load_time_s']:.2f}s")

    return {'documents': documents, 'source_stats': stats}

def run_loader(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta el agente de carga de documentos.
    """
    try:
        result = load_document(state)
        state["documents"] = result["documents"]
        state["source_stats"] = result["source_stats"]
        state.pop("file_path", None)
        return state
    except Exception as e:
        logger.error(f"Error en run_loader: {str(e)}")
        raise