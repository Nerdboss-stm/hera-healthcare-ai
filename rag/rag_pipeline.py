"""RAG Pipeline — Retrieval-Augmented Generation for clinical AI.

Combines the MedicalRetriever (semantic search) with the existing T5
summarizer to produce evidence-grounded clinical outputs with citations.
"""

from __future__ import annotations
import logging
from typing import Optional

from rag.knowledge_base import MedicalKnowledgeBase
from rag.retriever import MedicalRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-augmented generation pipeline for clinical tasks.

    1. Takes a clinical query or note
    2. Retrieves relevant medical knowledge from the vector store
    3. Augments the input with retrieved context
    4. Generates output using the T5 summarizer (or returns context for agents)
    """

    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase | None = None,
        retriever: MedicalRetriever | None = None,
        summarizer_fn=None,
    ):
        self._kb = knowledge_base or MedicalKnowledgeBase()
        self._retriever = retriever or MedicalRetriever()
        self._summarizer_fn = summarizer_fn
        self._initialized = False

    def initialize(self) -> None:
        """Build the FAISS index from the knowledge base."""
        if self._initialized:
            return
        texts = self._kb.get_texts()
        metadata = self._kb.get_metadata()
        self._retriever.index(texts, metadata)
        self._initialized = True
        logger.info("RAG pipeline initialized with %d documents", len(texts))

    def retrieve(
        self, query: str, top_k: int = 3, threshold: float = 0.3
    ) -> list[dict]:
        """Retrieve relevant medical knowledge for a query."""
        if not self._initialized:
            self.initialize()
        return self._retriever.retrieve(query, top_k=top_k, threshold=threshold)

    def augment_and_generate(
        self,
        clinical_note: str,
        top_k: int = 3,
        max_length: int = 150,
    ) -> dict:
        """RAG-augmented summarization: retrieve context, then generate.

        Returns:
            Dict with keys: summary, citations, retrieved_context
        """
        if not self._initialized:
            self.initialize()

        # Retrieve relevant context
        retrieved = self._retriever.retrieve(clinical_note, top_k=top_k)

        # Build augmented prompt
        context_block = "\n\n".join(
            f"[{r['source']}]: {r['text']}" for r in retrieved
        )
        augmented_input = (
            f"Medical context:\n{context_block}\n\n"
            f"Clinical note:\n{clinical_note}"
        )

        # Generate summary
        summary = None
        if self._summarizer_fn:
            try:
                summary = self._summarizer_fn(augmented_input, max_length=max_length)
            except Exception as e:
                logger.warning("Summarizer failed on augmented input: %s", e)
                # Fallback: summarize original note without RAG context
                try:
                    summary = self._summarizer_fn(clinical_note, max_length=max_length)
                except Exception:
                    summary = None

        citations = [
            {"source": r["source"], "relevance": r["score"]}
            for r in retrieved
        ]

        return {
            "summary": summary,
            "citations": citations,
            "retrieved_context": retrieved,
            "augmented": summary is not None,
        }

    def query_knowledge(self, question: str, top_k: int = 5) -> dict:
        """Query the medical knowledge base directly.

        Returns relevant passages with citations, without generation.
        """
        if not self._initialized:
            self.initialize()

        results = self._retriever.retrieve(question, top_k=top_k, threshold=0.2)
        return {
            "query": question,
            "results": results,
            "total_corpus_size": self._retriever.corpus_size,
        }

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._retriever.is_indexed
