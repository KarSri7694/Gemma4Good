from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
import warnings

from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.model_control import MODEL_CONTROL_PATH, load_model_sampling_config


ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "subject_materials"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "si", "the", "to", "was", "what", "when", "where", "which", "who",
    "why", "with",
}


def _get_chroma_collection():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError(
            "ChromaDB is not installed. Add the 'chromadb' dependency before using reading-material retrieval."
        ) from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _embedding_client() -> tuple[LlamaServerClient, str]:
    config = load_model_sampling_config(MODEL_CONTROL_PATH)
    return (
        LlamaServerClient(LlamaServerConfig(base_url=config.embedding_model_server)),
        config.embedding_model_name,
    )


def _warn_embedding_unavailable(exc: Exception) -> None:
    warnings.warn(
        f"Embedding server unavailable; continuing without vector retrieval: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


def _normalize_query_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", value.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _query_terms(value: str) -> list[str]:
    return [
        token
        for token in _normalize_query_text(value).split()
        if len(token) > 2 and token not in STOPWORDS
    ]


def _term_overlap_score(query_terms: list[str], haystack: str) -> float:
    if not query_terms:
        return 0.0
    normalized_haystack = _normalize_query_text(haystack)
    matches = sum(1 for token in query_terms if token in normalized_haystack)
    return matches / len(query_terms)


def _definition_intent(query: str) -> bool:
    normalized = _normalize_query_text(query)
    return (
        normalized.startswith("what is ")
        or normalized.startswith("define ")
        or normalized.startswith("meaning of ")
        or normalized.startswith("explain ")
    )


def _definition_chunk_boost(query_terms: list[str], heading: str, chapter_name: str, text: str) -> float:
    if not query_terms:
        return 0.0
    primary = query_terms[0]
    heading_text = _normalize_query_text(f"{heading} {chapter_name}")
    body_text = _normalize_query_text(text[:300])
    if primary and (heading_text.startswith(primary) or heading_text == primary):
        return 0.2
    if primary and body_text.startswith(f"{primary} is"):
        return 0.15
    return 0.0


def _rerank_hits(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = _query_terms(query)
    definition_query = _definition_intent(query)
    reranked: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for hit in hits:
        metadata = hit.get("metadata", {})
        text = str(hit.get("text", "") or "")
        dedupe_key = text.strip().lower()
        if not dedupe_key or dedupe_key in seen_texts:
            continue
        seen_texts.add(dedupe_key)
        section_heading = str(metadata.get("section_heading") or "")
        chapter_name = str(metadata.get("chapter_name") or "")
        source_title = str(metadata.get("source_title") or "")
        semantic_score = float(hit.get("score") or 0.0)
        lexical_score = _term_overlap_score(query_terms, f"{section_heading}\n{chapter_name}\n{text}")
        heading_score = _term_overlap_score(query_terms, f"{section_heading}\n{chapter_name}\n{source_title}")
        definition_boost = (
            _definition_chunk_boost(query_terms, section_heading, chapter_name, text)
            if definition_query
            else 0.0
        )
        blended_score = (
            (semantic_score * 0.55)
            + (lexical_score * 0.30)
            + (heading_score * 0.15)
            + definition_boost
        )
        reranked.append(
            {
                **hit,
                "score": round(blended_score, 4),
                "retrieval_debug": {
                    "semantic_score": round(semantic_score, 4),
                    "lexical_score": round(lexical_score, 4),
                    "heading_score": round(heading_score, 4),
                    "definition_boost": round(definition_boost, 4),
                },
            }
        )
    reranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return reranked


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client, model_name = _embedding_client()
    response = client.embeddings(input_text=texts, extra_payload={"model": model_name})
    data = response.get("data", [])
    embeddings = [item.get("embedding", []) for item in data if isinstance(item, dict)]
    if len(embeddings) != len(texts):
        raise ValueError("Embedding server returned an unexpected number of embeddings.")
    return embeddings


def try_embed_texts(texts: list[str]) -> tuple[list[list[float]], str | None]:
    if not texts:
        return [], None
    try:
        return embed_texts(texts), None
    except Exception as exc:
        _warn_embedding_unavailable(exc)
        return [], str(exc)


def index_material_chunks(
    *,
    source_material_id: int,
    grade: str,
    subject: str,
    chunks: list[dict[str, Any]],
) -> int:
    if not chunks:
        return 0
    collection = _get_chroma_collection()
    texts = [str(chunk.get("chunk_text", "")).strip() for chunk in chunks]
    embeddings, _error = try_embed_texts(texts)
    if len(embeddings) != len(texts):
        return 0
    ids = [f"material-{source_material_id}-chunk-{int(chunk.get('chunk_index', index))}" for index, chunk in enumerate(chunks)]
    metadatas = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.get("metadata", {}))
        metadata.update(
            {
                "source_material_id": source_material_id,
                "grade": str(grade),
                "subject": str(subject),
                "chunk_index": int(chunk.get("chunk_index", index)),
                "page_start": chunk.get("page_start") or 0,
                "page_end": chunk.get("page_end") or 0,
                "section_heading": str(chunk.get("section_heading") or ""),
                "content_type": str(chunk.get("content_type") or "text"),
            }
        )
        metadatas.append(metadata)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(ids)


def search_subject_materials(
    *,
    grade: str,
    subject: str,
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    normalized_query = query.strip()
    if not normalized_query:
        return []
    collection = _get_chroma_collection()
    config = load_model_sampling_config(MODEL_CONTROL_PATH)
    query_embeddings, _error = try_embed_texts([normalized_query])
    if not query_embeddings:
        return []
    query_embedding = query_embeddings[0]
    requested_top_k = max(1, top_k or config.rag_top_k)
    raw_candidate_count = max(requested_top_k * 4, 12)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=raw_candidate_count,
        where={"$and": [{"grade": str(grade)}, {"subject": str(subject)}]},
    )
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    items: list[dict[str, Any]] = []
    for item_id, document, metadata, distance in zip(ids, docs, metadatas, distances):
        score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
        items.append(
            {
                "id": item_id,
                "text": document,
                "metadata": metadata or {},
                "score": round(score, 4),
            }
        )
    reranked = _rerank_hits(normalized_query, items)
    return reranked[:requested_top_k]


def build_retrieval_context(
    *,
    grade: str,
    subject: str,
    query: str,
    top_k: int | None = None,
) -> str:
    hits = search_subject_materials(grade=grade, subject=subject, query=query, top_k=top_k)
    if not hits:
        return ""
    lines = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit.get("metadata", {})
        source_title = metadata.get("source_title") or f"Material {metadata.get('source_material_id', '')}"
        section_heading = metadata.get("section_heading") or "General"
        page_start = metadata.get("page_start") or ""
        page_end = metadata.get("page_end") or ""
        page_label = ""
        if page_start and page_end and page_start != page_end:
            page_label = f" pages {page_start}-{page_end}"
        elif page_start:
            page_label = f" page {page_start}"
        lines.append(
            f"[{index}] {source_title} | {section_heading}{page_label}\n{hit['text']}"
        )
    return "\n\n".join(lines)
