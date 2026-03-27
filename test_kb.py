import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from kb import (
    load_pdf_files,
    split_documents,
    build_vectorstore_from_pdfs,
    save_vectorstore,
    _enrich_metadata,
)


class TestEnrichMetadata:
    """Test metadata enrichment for documents."""

    def test_enrich_metadata_basic(self):
        """Test basic metadata enrichment."""
        docs = [
            Document(page_content="Test content", metadata={"source": "/path/to/test.pdf", "page": 0}),
        ]
        
        result = _enrich_metadata(docs, source_name="test.pdf")
        
        assert len(result) == 1
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["page"] == 1  # page should be incremented

    def test_enrich_metadata_page_increment(self):
        """Test that page numbers are incremented by 1."""
        docs = [
            Document(page_content="Content", metadata={"source": "file.pdf", "page": 2}),
        ]
        
        result = _enrich_metadata(docs)
        
        assert result[0].metadata["page"] == 3

    def test_enrich_metadata_multiple_docs(self):
        """Test enriching multiple documents."""
        docs = [
            Document(page_content="Content 1", metadata={"source": "doc.pdf", "page": 0}),
            Document(page_content="Content 2", metadata={"source": "doc.pdf", "page": 1}),
        ]
        
        result = _enrich_metadata(docs)
        
        assert len(result) == 2
        assert result[0].metadata["page"] == 1
        assert result[1].metadata["page"] == 2

    def test_enrich_metadata_handles_missing_page(self):
        """Test handling of missing page metadata."""
        docs = [
            Document(page_content="Content", metadata={"source": "file.pdf"}),
        ]
        
        result = _enrich_metadata(docs)
        
        assert result[0].metadata["page"] == 1


class TestSplitDocuments:
    """Test document splitting functionality."""

    def test_split_documents_basic(self):
        """Test basic document splitting."""
        large_content = "word " * 500  # Large content
        docs = [Document(page_content=large_content, metadata={"source": "test.pdf"})]
        
        result = split_documents(docs, chunk_size=100, chunk_overlap=20)
        
        assert len(result) > 1
        assert all(isinstance(doc, Document) for doc in result)

    def test_split_documents_preserves_metadata(self):
        """Test that splitting preserves document metadata."""
        doc = Document(
            page_content="word " * 300,
            metadata={"source": "test.pdf", "page": 1}
        )
        
        result = split_documents([doc])
        
        assert all(result[0].metadata["source"] == "test.pdf" for r in result)

    def test_split_documents_with_custom_chunk_size(self):
        """Test splitting with custom chunk size."""
        large_content = "word " * 300
        docs = [Document(page_content=large_content, metadata={"source": "test.pdf"})]
        
        result_small = split_documents(docs, chunk_size=50, chunk_overlap=10)
        result_large = split_documents(docs, chunk_size=200, chunk_overlap=50)
        
        assert len(result_small) > len(result_large)

    def test_split_documents_empty_list(self):
        """Test splitting empty document list."""
        result = split_documents([])
        assert result == []


class TestLoadPdfFiles:
    """Test PDF file loading."""

    @patch("kb.PyPDFLoader")
    def test_load_pdf_files_single(self, mock_loader_class):
        """Test loading a single PDF file."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = [
            Document(
                page_content="Page 1 content",
                metadata={"source": "test.pdf", "page": 0}
            )
        ]
        
        result = load_pdf_files(["test.pdf"])
        
        assert len(result) == 1
        assert result[0].page_content == "Page 1 content"
        assert result[0].metadata["source"] == "test.pdf"

    @patch("kb.PyPDFLoader")
    def test_load_pdf_files_multiple(self, mock_loader_class):
        """Test loading multiple PDF files."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Return different content for each call
        mock_loader.load.side_effect = [
            [Document(page_content="PDF1", metadata={"source": "file1.pdf", "page": 0})],
            [Document(page_content="PDF2", metadata={"source": "file2.pdf", "page": 0})],
        ]
        
        result = load_pdf_files(["file1.pdf", "file2.pdf"])
        
        assert len(result) == 2
        assert mock_loader_class.call_count == 2

    @patch("kb.PyPDFLoader")
    def test_load_pdf_files_empty_list(self, mock_loader_class):
        """Test loading empty list of PDFs."""
        result = load_pdf_files([])
        assert result == []


class TestBuildVectorstoreFromPdfs:
    """Test vectorstore building from PDFs."""

    @patch("kb.get_embedding_model")
    @patch("kb.FAISS.from_documents")
    @patch("kb.PyPDFLoader")
    def test_build_vectorstore_from_pdfs(self, mock_loader_class, mock_faiss, mock_embeddings):
        """Test building vectorstore from PDFs."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = [
            Document(page_content="content " * 100, metadata={"source": "test.pdf", "page": 0})
        ]
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_faiss.return_value = mock_vectorstore
        
        vectorstore, chunks = build_vectorstore_from_pdfs(["test.pdf"])
        
        assert vectorstore is not None
        assert len(chunks) > 0
        mock_embeddings.assert_called_once()
        mock_faiss.assert_called_once()

    @patch("kb.get_embedding_model")
    @patch("kb.FAISS.from_documents")
    @patch("kb.PyPDFLoader")
    def test_build_vectorstore_custom_embedding_model(self, mock_loader_class, mock_faiss, mock_embeddings):
        """Test building vectorstore with custom embedding model."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.load.return_value = [
            Document(page_content="content " * 100, metadata={"source": "test.pdf", "page": 0})
        ]
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_vectorstore = MagicMock()
        mock_faiss.return_value = mock_vectorstore
        
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        build_vectorstore_from_pdfs(["test.pdf"], embedding_model=custom_model)
        
        mock_embeddings.assert_called_with(custom_model)


class TestSaveVectorstore:
    """Test vectorstore saving."""

    @patch("kb.Path.mkdir")
    def test_save_vectorstore(self, mock_mkdir):
        """Test saving vectorstore."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.save_local = MagicMock()
        
        save_vectorstore(mock_vectorstore, "test_index")
        
        mock_mkdir.assert_called_once()
        mock_vectorstore.save_local.assert_called_once()

    @patch("kb.Path.mkdir")
    def test_save_vectorstore_creates_directory(self, mock_mkdir):
        """Test that save_vectorstore creates directory."""
        mock_vectorstore = MagicMock()
        
        save_vectorstore(mock_vectorstore, "nonexistent/path")
        
        # Verify mkdir was called with parents=True and exist_ok=True
        call_kwargs = mock_mkdir.call_args[1]
        assert call_kwargs["parents"] is True
        assert call_kwargs["exist_ok"] is True
