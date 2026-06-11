import os
import logging
from typing import List, Optional

logger = logging.getLogger("cqa_ai.rag")


class RAGStore:
    """
    Quản lý vector store FAISS cho RAG (Retrieval-Augmented Generation).
    Dùng để tìm kiếm ngữ nghĩa trong tài liệu nội bộ (PDF, TXT, ...).
    """

    def __init__(self):
        self._vectorstore = None
        self._embeddings = None
        self._docs_path = os.getenv("DOCS_PATH", "./data/docs")
        self._knowledge_path = os.getenv("KNOWLEDGE_PATH", "./data/knowledge")
        self._index_path = "./data/faiss_index"

    def load(self) -> bool:
        """Load FAISS index đã được build sẵn từ disk."""
        if not os.path.exists(self._index_path):
            logger.warning("Chưa có FAISS index tại %s. Hãy chạy scripts/index_knowledge.py", self._index_path)
            return False
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = self._build_embeddings()
            self._vectorstore = FAISS.load_local(
                self._index_path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Đã load FAISS index từ %s", self._index_path)
            return True
        except Exception as e:
            logger.error("Lỗi load FAISS index: %s", e)
            return False

    def search(self, query: str, k: int = 4) -> List[dict]:
        """Tìm kiếm ngữ nghĩa trong knowledge base."""
        if self._vectorstore is None:
            return []
        try:
            docs = self._vectorstore.similarity_search(query, k=k)
            return [
                {"title": doc.metadata.get("source", "Tài liệu"), "snippet": doc.page_content[:300]}
                for doc in docs
            ]
        except Exception as e:
            logger.error("Lỗi tìm kiếm RAG: %s", e)
            return []

    def _build_embeddings(self):
        """Tạo embedding model dựa trên API key."""
        deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if openai_key:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=openai_key)
        elif deepseek_key:
            # DeepSeek dùng OpenAI-compatible embedding
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                api_key=deepseek_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model="text-embedding-ada-002",
            )
        raise RuntimeError("Cần OPENAI_API_KEY hoặc DEEPSEEK_API_KEY để dùng RAG")
