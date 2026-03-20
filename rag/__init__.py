"""RAG-Powered Medical Knowledge Base.

Provides retrieval-augmented generation by indexing medical knowledge
into a FAISS vector store and retrieving relevant context to ground
clinical AI outputs in evidence-based literature.
"""

from rag.knowledge_base import MedicalKnowledgeBase
from rag.retriever import MedicalRetriever
from rag.rag_pipeline import RAGPipeline

__all__ = ["MedicalKnowledgeBase", "MedicalRetriever", "RAGPipeline"]
