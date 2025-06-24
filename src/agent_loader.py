import time
import os
import json
import logging
import datetime
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from typing import Dict, Any
# NUEVO: para metadatos nativos
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    # Limpieza avanzada: elimina caracteres no ASCII, normaliza espacios y saltos de línea
    text = re.sub(r'[\r\f\x0b]', ' ', text)  # elimina retornos de carro y form feeds
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)  # elimina espacios unicode invisibles
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # elimina caracteres no ASCII
    text = re.sub(r'\s+', ' ', text)  # normaliza espacios
    text = re.sub(r'\n+', '\n', text)  # normaliza saltos de línea
    return text.strip()

def extract_pdf_metadata(path):
    try:
        reader = PdfReader(path)
        meta = reader.metadata or {}
        return {
            'author': meta.get('/Author'),
            'title_pdf': meta.get('/Title'),
            'creation_date': meta.get('/CreationDate'),
            'producer': meta.get('/Producer'),
            'creator': meta.get('/Creator'),
            'moddate': meta.get('/ModDate')
        }
    except Exception as e:
        logging.warning(f"No se pudieron extraer metadatos PDF: {e}")
        return {}

def extract_docx_metadata(path):
    try:
        doc = DocxDocument(path)
        core = doc.core_properties
        return {
            'author': core.author,
            'title_docx': core.title,
            'created': str(core.created),
            'last_modified_by': core.last_modified_by,
            'last_printed': str(core.last_printed),
            'modified': str(core.modified),
            'category': core.category,
            'comments': core.comments,
            'subject': core.subject
        }
    except Exception as e:
        logging.warning(f"No se pudieron extraer metadatos DOCX: {e}")
        return {}

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
            meta_extra = {}
            if ext == '.pdf':
                # Cargar PDF
                loader = PyPDFLoader(path)
                pages = loader.load()
                
                # Combinar todas las páginas en un solo documento
                combined_text = "\n".join(page.page_content for page in pages)
                combined_text = clean_text(combined_text)
                
                # Extraer metadatos del PDF
                meta_extra = extract_pdf_metadata(path)
                
                # Crear un solo documento con el texto combinado
                documents.append({
                    'title': os.path.splitext(os.path.basename(path))[0],
                    'text': combined_text,
                    'metadata': {
                        'source': path,
                        'total_pages': len(pages),
                        **meta_extra
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
                
                content = clean_text(content)
                documents.append({
                    'title': os.path.splitext(os.path.basename(path))[0],
                    'text': content,
                    'metadata': {
                        'source': path,
                        'total_pages': 1
                    }
                })
                total_pages += 1
                
            elif ext in ['.docx', '.doc']:
                # Cargar otros tipos de documentos
                loader = UnstructuredWordDocumentLoader(path)
                pages = loader.load()
                text = "".join(p.page_content for p in pages)
                text = clean_text(text)
                
                meta_extra = extract_docx_metadata(path)
                documents.append({
                    'title': os.path.splitext(os.path.basename(path))[0],
                    'text': text,
                    'metadata': {
                        'source': path,
                        'total_pages': len(pages),
                        **meta_extra
                    }
                })
                total_pages += len(pages)
            
            else:
                # Otros tipos: solo limpieza
                loader = UnstructuredWordDocumentLoader(path)
                pages = loader.load()
                text = clean_text("".join(p.page_content for p in pages))
                
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