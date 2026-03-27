"""
Pytest configuration and shared fixtures for RAG tests.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from dotenv import load_dotenv


# Load environment variables for testing
load_dotenv()


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables between tests."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore for testing."""
    vectorstore = MagicMock()
    vectorstore.as_retriever = MagicMock()
    vectorstore.save_local = MagicMock()
    return vectorstore


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.invoke = MagicMock()
    return llm


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model for testing."""
    embeddings = MagicMock()
    embeddings.embed_query = MagicMock(return_value=[0.1] * 384)
    embeddings.embed_documents = MagicMock(return_value=[[0.1] * 384])
    return embeddings


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="This is the first document about machine learning and AI.",
            metadata={"source": "doc1.pdf", "page": 1}
        ),
        Document(
            page_content="This is the second document about natural language processing.",
            metadata={"source": "doc2.pdf", "page": 1}
        ),
        Document(
            page_content="This is the third document about vector databases.",
            metadata={"source": "doc3.pdf", "page": 2}
        ),
    ]


@pytest.fixture
def temp_index_dir(tmp_path):
    """Create a temporary index directory."""
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()
    return str(index_dir)


@pytest.fixture
def temp_pdf_dir(tmp_path):
    """Create a temporary PDF directory."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    return str(pdf_dir)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external services)"
    )


# Optional: Configure test collection
pytest_plugins = []
