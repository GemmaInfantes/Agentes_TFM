from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict, Annotated, Literal
import operator

# Funciones de merge para campos del estado:

def update_file_path(
    existing: Optional[str] = None,
    updates: Optional[Union[str, Literal["clear"]]] = None,
) -> Optional[str]:
    """
    Controla cómo se fusiona el campo `file_path`.
    - Si `existing` es None o cadena vacía, devuelve `updates`.
    - Si `existing` ya está relleno, lo conserva (ignora `updates`).
    - Si `updates` == "clear", devuelve cadena vacía.
    """
    if existing is None:
        existing = ""
    if updates is None:
        return existing
    if updates == "clear":
        return ""
    if existing:
        return existing
    return updates


def update_documents(
    existing: Optional[List[Dict[str, Any]]] = None,
    updates: Optional[Union[List[Dict[str, Any]], Literal["clear"]]] = None,
) -> List[Dict[str, Any]]:
    """
    Controla cómo se fusiona el campo `documents`.
    - Si `existing` es None, lo inicializa a lista vacía.
    - Si `updates` es None, devuelve `existing`.
    - Si `updates` == "clear", devuelve lista vacía.
    - Si `existing` ya tenía contenido, lo conserva.
    """
    if existing is None:
        existing = []
    if updates is None:
        return existing
    if updates == "clear":
        return []
    if existing:
        return existing
    return updates


def update_source_stats(
    existing: Optional[Dict[str, Any]] = None,
    updates: Optional[Union[Dict[str, Any], Literal["clear"]]] = None,
) -> Dict[str, Any]:
    """
    Controla cómo se fusiona el campo `source_stats`.
    - Si `existing` es None, lo inicializa a dict vacío.
    - Si `updates` es None, devuelve `existing`.
    - Si `updates` == "clear", devuelve dict vacío.
    - Si `existing` ya tenía contenido, lo conserva.
    """
    if existing is None:
        existing = {}
    if updates is None:
        return existing
    if updates == "clear":
        return {}
    if existing:
        return existing
    return updates


def update_metadatos(
    existing: Optional[List[Dict[str, Any]]] = None,
    updates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Actualiza los metadatos de los documentos.
    Si no hay metadatos existentes, crea una nueva lista.
    Si hay actualizaciones, las aplica a los metadatos existentes.
    """
    if existing is None:
        existing = []
    
    if updates is None:
        return existing
    
    # Asegurar que la longitud de los metadatos coincida con el número de documentos
    while len(existing) < len(updates):
        existing.append({})
    
    # Actualizar los metadatos existentes con las actualizaciones
    for i, update in enumerate(updates):
        if i < len(existing):
            # Si el documento ya tiene metadatos, actualizarlos
            if existing[i]:
                existing[i].update(update)
            else:
                # Si no tiene metadatos, asignar los nuevos directamente
                existing[i] = update
    
    return existing


class DocState(TypedDict, total=False):
    # 1) Entrada inicial:
    file_path: Annotated[Optional[str], update_file_path]
    # 2) Tras LoaderAgent y MetadataAgent:
    documents: Annotated[List[Dict[str, Any]], update_documents]
    source_stats: Annotated[Dict[str, Any], update_source_stats]
    # 3) Paralelismo: Summaries, Keywords, Topics, Structure, Insights (merge operator.add)
    metadatos: Annotated[List[Dict[str, Any]], update_metadatos]
    embeddings: Annotated[List[List[float]], operator.add]
    # 4) Resultado final:
    index_result: Annotated[Dict[str, Any], operator.or_]