"""
Script build FAISS index từ các tài liệu trong data/knowledge/ và data/docs/

Chạy: python scripts/index_knowledge.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def index_knowledge():
    from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    docs_path = os.getenv("DOCS_PATH", "./data/docs")
    knowledge_path = os.getenv("KNOWLEDGE_PATH", "./data/knowledge")
    index_path = "./data/faiss_index"

    # Xây dựng embeddings
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(api_key=openai_key)
    elif deepseek_key:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            api_key=deepseek_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model="text-embedding-ada-002",
        )
    else:
        print("❌ Cần OPENAI_API_KEY hoặc DEEPSEEK_API_KEY để tạo embeddings")
        sys.exit(1)

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Load PDF từ data/docs
    for folder in [docs_path, knowledge_path]:
        if not os.path.exists(folder):
            continue
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if filename.endswith(".pdf"):
                loader = PyMuPDFLoader(filepath)
                docs = loader.load()
                all_docs.extend(splitter.split_documents(docs))
                print(f"  ✅ PDF: {filename} ({len(docs)} trang)")
            elif filename.endswith(".txt") or filename.endswith(".md"):
                loader = TextLoader(filepath, encoding="utf-8")
                docs = loader.load()
                all_docs.extend(splitter.split_documents(docs))
                print(f"  ✅ Text: {filename}")

    if not all_docs:
        print("⚠️  Không tìm thấy tài liệu nào để index")
        return

    print(f"\n📚 Tổng: {len(all_docs)} chunks — đang build FAISS index...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Đã lưu FAISS index tại: {index_path}")


if __name__ == "__main__":
    index_knowledge()
