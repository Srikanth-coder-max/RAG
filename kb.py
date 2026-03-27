from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from rag_core import get_embedding_model


def _enrich_metadata(documents, source_name: str | None = None):
    for d in documents:
        original_source = d.metadata.get("source", source_name or "unknown")
        source_path = Path(original_source)
        d.metadata["source"] = source_path.name
        d.metadata["source_path"] = str(source_path)
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
    return documents


def load_pdf_files(pdf_paths: Iterable[str]):
    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        loaded = loader.load()
        docs.extend(_enrich_metadata(loaded, source_name=Path(pdf_path).name))
    return docs


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def build_vectorstore_from_pdfs(pdf_paths: Iterable[str], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    documents = load_pdf_files(pdf_paths)
    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    embeddings = get_embedding_model(embedding_model)
    return FAISS.from_documents(chunks, embeddings), chunks


def save_vectorstore(vectorstore: FAISS, index_dir: str):
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))
