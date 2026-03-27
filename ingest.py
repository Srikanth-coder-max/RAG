import argparse
from pathlib import Path

from kb import build_vectorstore_from_pdfs, save_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs and build a local FAISS index.")
    parser.add_argument("--pdf-folder", type=str, default="data", help="Folder containing PDF files")
    parser.add_argument("--index-dir", type=str, default="faiss_index", help="Output FAISS index directory")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name",
    )
    args = parser.parse_args()

    pdf_folder = Path(args.pdf_folder)
    index_dir = Path(args.index_dir)
    pdf_files = sorted(str(p) for p in pdf_folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_folder}")

    vectorstore, chunks = build_vectorstore_from_pdfs(pdf_files, embedding_model=args.embedding_model)
    save_vectorstore(vectorstore, str(index_dir))
    print(f"Indexed {len(chunks)} chunks into: {index_dir}")


if __name__ == "__main__":
    main()
