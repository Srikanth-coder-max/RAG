import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from rag_core import (
    get_embedding_model,
    load_vectorstore,
    get_llm,
    build_retrieval_qa_chain,
    format_citations,
)


class TestGetEmbeddingModel:
    """Test embedding model initialization."""

    @patch("rag_core.HuggingFaceEmbeddings")
    def test_get_embedding_model_default(self, mock_hf_embeddings):
        """Test getting default embedding model."""
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings
        
        result = get_embedding_model()
        
        mock_hf_embeddings.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        assert result is mock_embeddings

    @patch("rag_core.HuggingFaceEmbeddings")
    def test_get_embedding_model_custom(self, mock_hf_embeddings):
        """Test getting embedding model with custom name."""
        mock_embeddings = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings
        
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        result = get_embedding_model(custom_model)
        
        mock_hf_embeddings.assert_called_once_with(model_name=custom_model)
        assert result is mock_embeddings


class TestLoadVectorstore:
    """Test vectorstore loading."""

    @patch("rag_core.get_embedding_model")
    @patch("rag_core.FAISS.load_local")
    def test_load_vectorstore(self, mock_faiss_load, mock_get_embeddings):
        """Test loading vectorstore."""
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectorstore = MagicMock()
        mock_faiss_load.return_value = mock_vectorstore
        
        result = load_vectorstore("/path/to/index")
        
        mock_get_embeddings.assert_called_once()
        mock_faiss_load.assert_called_once_with(
            "/path/to/index",
            mock_embeddings,
            allow_dangerous_deserialization=True
        )
        assert result is mock_vectorstore

    @patch("rag_core.get_embedding_model")
    @patch("rag_core.FAISS.load_local")
    def test_load_vectorstore_custom_embedding(self, mock_faiss_load, mock_get_embeddings):
        """Test loading vectorstore with custom embedding model."""
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        mock_faiss_load.return_value = MagicMock()
        
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        load_vectorstore("/path/to/index", embedding_model=custom_model)
        
        mock_get_embeddings.assert_called_once_with(custom_model)


class TestGetLlm:
    """Test LLM initialization."""

    @patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o-mini"})
    @patch("rag_core.ChatOpenAI")
    def test_get_llm_openai_default(self, mock_chat_openai):
        """Test getting OpenAI LLM with defaults."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        result = get_llm(provider="openai")
        
        mock_chat_openai.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.0
        )
        assert result is mock_llm

    @patch("rag_core.ChatOpenAI")
    def test_get_llm_openai_custom_model(self, mock_chat_openai):
        """Test getting OpenAI LLM with custom model."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        result = get_llm(provider="openai", model="gpt-4-turbo", temperature=0.5)
        
        mock_chat_openai.assert_called_once_with(
            model="gpt-4-turbo",
            temperature=0.5
        )

    @patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"})
    @patch("rag_core.ChatGoogleGenerativeAI")
    def test_get_llm_gemini_default(self, mock_chat_gemini):
        """Test getting Gemini LLM with defaults."""
        mock_llm = MagicMock()
        mock_chat_gemini.return_value = mock_llm
        
        result = get_llm(provider="gemini")
        
        mock_chat_gemini.assert_called_once_with(
            model="gemini-2.5-flash",
            temperature=0.0
        )
        assert result is mock_llm

    @patch("rag_core.ChatGoogleGenerativeAI")
    def test_get_llm_gemini_deprecated_model(self, mock_chat_gemini):
        """Test that deprecated Gemini models are handled."""
        mock_llm = MagicMock()
        mock_chat_gemini.return_value = mock_llm
        
        # Test with deprecated model name
        result = get_llm(provider="gemini", model="gemini-1.5-flash")
        
        mock_chat_gemini.assert_called_once_with(
            model="gemini-2.5-flash",  # Should be replaced
            temperature=0.0
        )

    @patch.dict(os.environ, {
        "BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "AWS_REGION": "us-west-2"
    })
    @patch("rag_core.ChatBedrock")
    def test_get_llm_bedrock_default(self, mock_chat_bedrock):
        """Test getting Bedrock LLM with defaults."""
        mock_llm = MagicMock()
        mock_chat_bedrock.return_value = mock_llm
        
        result = get_llm(provider="bedrock")
        
        mock_chat_bedrock.assert_called_once_with(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name="us-west-2",
            model_kwargs={"temperature": 0.0}
        )
        assert result is mock_llm

    @patch("rag_core.ChatBedrock")
    def test_get_llm_bedrock_custom_model(self, mock_chat_bedrock):
        """Test getting Bedrock LLM with custom model."""
        mock_llm = MagicMock()
        mock_chat_bedrock.return_value = mock_llm
        
        result = get_llm(
            provider="bedrock",
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0.3
        )
        
        call_kwargs = mock_chat_bedrock.call_args[1]
        assert call_kwargs["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert call_kwargs["model_kwargs"]["temperature"] == 0.3

    def test_get_llm_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="provider must be one of"):
            get_llm(provider="invalid_provider")

    @patch.dict(os.environ, {"BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20240620-v1:0"})
    @patch("rag_core.ChatBedrock")
    def test_get_llm_bedrock_default_region(self, mock_chat_bedrock):
        """Test Bedrock defaults to us-east-1 region."""
        mock_llm = MagicMock()
        mock_chat_bedrock.return_value = mock_llm
        
        get_llm(provider="bedrock")
        
        call_kwargs = mock_chat_bedrock.call_args[1]
        assert call_kwargs["region_name"] == "us-east-1"


class TestBuildRetrievalQaChain:
    """Test RAG chain building."""

    def test_build_retrieval_qa_chain_basic(self):
        """Test basic RAG chain construction."""
        # Create mock vectorstore with as_retriever method
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        # Create mock LLM
        mock_llm = MagicMock()
        
        result = build_retrieval_qa_chain(mock_vectorstore, mock_llm)
        
        # Verify as_retriever was called
        mock_vectorstore.as_retriever.assert_called_once()
        assert result is not None

    def test_build_retrieval_qa_chain_custom_k(self):
        """Test building chain with custom k parameter."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_llm = MagicMock()
        
        build_retrieval_qa_chain(mock_vectorstore, mock_llm, k=8)
        
        # Get the search_kwargs passed to as_retriever
        call_args = mock_vectorstore.as_retriever.call_args[1]
        assert call_args["search_kwargs"]["k"] == 8

    def test_build_retrieval_qa_chain_with_source_filters(self):
        """Test building chain with source filters."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_llm = MagicMock()
        
        source_filters = ["document1.pdf", "document2.pdf"]
        build_retrieval_qa_chain(mock_vectorstore, mock_llm, source_filters=source_filters)
        
        # Verify filter was set
        call_args = mock_vectorstore.as_retriever.call_args[1]
        assert "filter" in call_args["search_kwargs"]

    def test_build_retrieval_qa_chain_source_filter_function(self):
        """Test that source filter function works correctly."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_llm = MagicMock()
        
        source_filters = ["test.pdf"]
        build_retrieval_qa_chain(mock_vectorstore, mock_llm, source_filters=source_filters)
        
        # Get the filter function
        call_args = mock_vectorstore.as_retriever.call_args[1]
        filter_func = call_args["search_kwargs"]["filter"]
        
        # Test the filter function
        assert filter_func({"source": "test.pdf"}) is True
        assert filter_func({"source": "other.pdf"}) is False

    def test_build_retrieval_qa_chain_without_source_filters(self):
        """Test that no filter is added when source_filters is None."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_llm = MagicMock()
        
        build_retrieval_qa_chain(mock_vectorstore, mock_llm, source_filters=None)
        
        call_args = mock_vectorstore.as_retriever.call_args[1]
        assert "filter" not in call_args["search_kwargs"]

    def test_build_retrieval_qa_chain_empty_source_filters(self):
        """Test that empty source_filters list is treated as no filters."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_llm = MagicMock()
        
        build_retrieval_qa_chain(mock_vectorstore, mock_llm, source_filters=[])
        
        call_args = mock_vectorstore.as_retriever.call_args[1]
        # Empty list should not add filter
        assert "filter" not in call_args["search_kwargs"]

    def test_build_retrieval_qa_chain_returns_runnable(self):
        """Test that the returned chain is a valid runnable."""
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_llm = MagicMock()
        
        result = build_retrieval_qa_chain(mock_vectorstore, mock_llm)
        
        # Result should have invoke/batch methods (Runnable interface)
        assert hasattr(result, "invoke") or hasattr(result, "__or__")


class TestFormatCitations:
    """Test citation formatting."""

    def test_format_citations_empty(self):
        """Test formatting empty context."""
        result = format_citations([])
        assert result == []

    def test_format_citations_none(self):
        """Test formatting None context."""
        result = format_citations(None)
        assert result == []

    def test_format_citations_single_document(self):
        """Test formatting single document citation."""
        doc = Document(
            page_content="This is the document content.",
            metadata={"source": "test.pdf", "page": 1}
        )
        
        result = format_citations([doc])
        
        assert len(result) == 1
        assert result[0]["source"] == "test.pdf"
        assert result[0]["page"] == 1
        assert "document content" in result[0]["snippet"]

    def test_format_citations_multiple_documents(self):
        """Test formatting multiple document citations."""
        docs = [
            Document(
                page_content="First document content here.",
                metadata={"source": "doc1.pdf", "page": 1}
            ),
            Document(
                page_content="Second document content here.",
                metadata={"source": "doc2.pdf", "page": 2}
            ),
        ]
        
        result = format_citations(docs)
        
        assert len(result) == 2
        assert result[0]["source"] == "doc1.pdf"
        assert result[1]["source"] == "doc2.pdf"

    def test_format_citations_long_content_truncated(self):
        """Test that long content is truncated to 220 chars."""
        long_content = "word " * 100  # Very long content
        doc = Document(
            page_content=long_content,
            metadata={"source": "test.pdf", "page": 1}
        )
        
        result = format_citations([doc])
        
        # Snippet should be truncated to approximately 220 chars
        assert len(result[0]["snippet"]) <= 230  # Allow some margin
        
    def test_format_citations_missing_metadata(self):
        """Test handling of missing metadata fields."""
        doc = Document(
            page_content="Content without full metadata.",
            metadata={}
        )
        
        result = format_citations([doc])
        
        assert result[0]["source"] == "unknown"
        assert result[0]["page"] == "?"

    def test_format_citations_preserves_order(self):
        """Test that citation order is preserved."""
        docs = [
            Document(
                page_content="First",
                metadata={"source": f"doc{i}.pdf", "page": i}
            )
            for i in range(5)
        ]
        
        result = format_citations(docs)
        
        for i, citation in enumerate(result):
            assert citation["source"] == f"doc{i}.pdf"

    def test_format_citations_snippet_contains_content(self):
        """Test that snippet is derived from page_content."""
        content = "This is important information about the topic."
        doc = Document(
            page_content=content,
            metadata={"source": "test.pdf", "page": 1}
        )
        
        result = format_citations([doc])
        
        # The snippet should contain key words from the content
        assert "important" in result[0]["snippet"]
