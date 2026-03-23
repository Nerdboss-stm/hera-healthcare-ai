"""Medical Retriever — FAISS-based semantic search over medical knowledge.

Embeds medical knowledge chunks using sentence-transformers and indexes
them in FAISS for fast similarity retrieval. Returns top-k relevant
passages with source citations and similarity scores.
"""

from __future__ import annotations
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports to avoid heavy dependencies at module load
_faiss = None
_SentenceTransformer = None


_lightweight = os.getenv("HERA_LIGHTWEIGHT", "").lower() in ("1", "true", "yes")


def _load_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss

            _faiss = faiss
        except ImportError:
            logger.warning("faiss not installed, using keyword fallback")
            return None
    return _faiss


def _load_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer

            _SentenceTransformer = SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, using keyword fallback"
            )
            return None
    return _SentenceTransformer


class MedicalRetriever:
    """Semantic search engine over a medical knowledge corpus.

    Uses sentence-transformers to encode texts into dense vectors,
    then indexes in FAISS for sub-millisecond retrieval.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        self._model_name = model_name
        self._dimension = dimension
        self._encoder = None
        self._index = None
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    def _ensure_encoder(self):
        if self._encoder is None:
            ST = _load_sentence_transformer()
            self._encoder = ST(self._model_name)
        return self._encoder

    def index(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Embed and index a corpus of texts."""
        self._texts = list(texts)
        self._metadata = metadata or [{} for _ in texts]

        faiss = _load_faiss()
        if faiss is None or _lightweight:
            logger.info("Using keyword fallback for %d documents", len(texts))
            self._index = "keyword"
            return

        encoder = self._ensure_encoder()
        if encoder is None:
            self._index = "keyword"
            return

        logger.info("Indexing %d documents into FAISS", len(texts))
        embeddings = encoder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

        logger.info(
            "Indexed %d vectors (dim=%d)", self._index.ntotal, embeddings.shape[1]
        )

    def _keyword_retrieve(self, query: str, top_k: int) -> list[dict]:
        """Simple keyword-based retrieval fallback."""
        query_terms = set(query.lower().split())
        scored = []
        for i, text in enumerate(self._texts):
            text_lower = text.lower()
            hits = sum(1 for t in query_terms if t in text_lower)
            if hits > 0:
                score = hits / max(len(query_terms), 1)
                scored.append((score, i))
        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            results.append(
                {
                    "text": self._texts[idx],
                    "score": round(score, 4),
                    "source": meta.get("source", "unknown"),
                    "metadata": meta,
                }
            )
        return results

    def retrieve(
        self, query: str, top_k: int = 3, threshold: float = 0.3
    ) -> list[dict]:
        """Retrieve top-k relevant passages for a query.

        Returns:
            List of dicts with keys: text, score, source, metadata
        """
        if self._index is None:
            logger.warning("Retriever index is empty")
            return []

        if self._index == "keyword":
            return self._keyword_retrieve(query, top_k)

        if self._index.ntotal == 0:
            return []

        encoder = self._ensure_encoder()
        query_vec = encoder.encode([query], show_progress_bar=False)
        query_vec = np.array(query_vec, dtype="float32")
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            results.append(
                {
                    "text": self._texts[idx],
                    "score": round(float(score), 4),
                    "source": meta.get("source", "unknown"),
                    "metadata": meta,
                }
            )

        return results

    @property
    def is_indexed(self) -> bool:
        if self._index == "keyword":
            return len(self._texts) > 0
        return self._index is not None and self._index.ntotal > 0

    @property
    def corpus_size(self) -> int:
        if self._index == "keyword":
            return len(self._texts)
        return self._index.ntotal if self._index else 0
