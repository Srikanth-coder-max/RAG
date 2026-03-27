# RAG Unit Testing Guide

This document explains how to run and understand the unit tests for the RAG (Retrieval-Augmented Generation) PDF Chat application.

## Test Structure

The test suite is organized into separate test modules for each application component:

### Test Files

- **test_kb.py** - Tests for knowledge base operations
  - PDF file loading
  - Document splitting
  - Metadata enrichment
  - Vectorstore building and saving

- **test_rag_core.py** - Tests for RAG core functionality
  - Embedding model initialization
  - LLM provider setup (OpenAI, Gemini, Bedrock)
  - Vectorstore loading
  - RAG chain building
  - Citation formatting

- **test_ingest.py** - Tests for PDF ingestion pipeline
  - CLI argument parsing
  - PDF discovery and processing
  - Vectorstore persistence

- **test_app.py** - Tests for Streamlit UI utilities
  - Chat history formatting
  - Chat export functionality
  - File I/O operations

- **conftest.py** - Shared pytest fixtures and configuration
  - Mock objects (vectorstore, LLM, embeddings)
  - Sample test data
  - Environment setup/teardown

## Installation

Install testing dependencies:

```bash
pip install pytest pytest-mock
```

Or install from requirements:

```bash
pip install -r requirements.txt
pip install pytest pytest-mock
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest test_kb.py
```

### Run specific test class

```bash
pytest test_rag_core.py::TestGetLlm
```

### Run specific test function

```bash
pytest test_kb.py::TestSplitDocuments::test_split_documents_basic
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage reporting

```bash
pip install pytest-cov
pytest --cov=. --cov-report=html
```

### Run tests matching a pattern

```bash
pytest -k "embedding"
```

### Run tests excluding slow tests

```bash
pytest -m "not slow"
```

## Test Coverage Areas

### Knowledge Base Module (kb.py)
- ✅ PDF file loading and parsing
- ✅ Document splitting with configurable chunk sizes
- ✅ Metadata enrichment (source, page numbers)
- ✅ FAISS vectorstore creation
- ✅ Vectorstore persistence

### RAG Core Module (rag_core.py)
- ✅ Embedding model initialization (HuggingFace)
- ✅ LLM initialization (OpenAI, Gemini, Bedrock)
- ✅ Deprecation handling for Gemini models
- ✅ Vectorstore loading from disk
- ✅ RAG chain construction with retrieval
- ✅ Source filtering in retrieval
- ✅ Citation formatting from context

### Ingest Pipeline (ingest.py)
- ✅ CLI argument parsing
- ✅ PDF discovery in folders
- ✅ Error handling for missing PDFs
- ✅ Vectorstore building and saving
- ✅ Status reporting

### Streamlit App (app.py)
- ✅ Chat history formatting
- ✅ Chat export (Markdown, JSON)
- ✅ Directory creation
- ✅ File upload handling
- ✅ Vectorstore persistence

## Mocking Strategy

Tests use mocking extensively to avoid external dependencies:

- **External APIs**: LLMs (OpenAI, Gemini, Bedrock) are mocked
- **File System**: Uses temporary directories where needed
- **PDF Loading**: PyPDFLoader is mocked to return test documents
- **FAISS**: Vectorstore operations are mocked

This allows fast, reliable tests without needing API keys or actual PDFs.

## Test Examples

### Example: Testing PDF loading

```python
@patch("kb.PyPDFLoader")
def test_load_pdf_files_single(mock_loader_class):
    # Setup mock
    mock_loader = MagicMock()
    mock_loader_class.return_value = mock_loader
    mock_loader.load.return_value = [Document(...)]
    
    # Execute
    result = load_pdf_files(["test.pdf"])
    
    # Assert
    assert len(result) == 1
    assert result[0].page_content == "..."
```

### Example: Testing LLM initialization

```python
@patch("rag_core.ChatOpenAI")
def test_get_llm_openai_default(mock_chat_openai):
    # Setup mock
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm
    
    # Execute
    result = get_llm(provider="openai")
    
    # Assert
    mock_chat_openai.assert_called_once_with(model="...", temperature=0.0)
```

## Fixtures

Available fixtures in conftest.py:

- **mock_vectorstore** - Mocked FAISS vectorstore
- **mock_llm** - Mocked language model
- **mock_embeddings** - Mocked embedding model
- **sample_documents** - Pre-created test documents
- **temp_index_dir** - Temporary FAISS index directory
- **temp_pdf_dir** - Temporary PDF directory
- **reset_environment** - Resets environment variables between tests

### Using fixtures in tests

```python
def test_something(mock_vectorstore, sample_documents):
    # Use fixtures here
    vectorstore = mock_vectorstore
    docs = sample_documents
```

## Continuous Integration

To run tests in CI/CD pipelines:

```bash
# Run with coverage and generate reports
pytest --cov=. --cov-report=xml --cov-report=term

# Run with specific markers
pytest -m "not integration"

# Run with exit code on failure
pytest --tb=short || exit 1
```

## Extending Tests

To add new tests:

1. Create or update test file (test_*.py pattern)
2. Define test functions with `test_` prefix
3. Use existing fixtures or create new ones in conftest.py
4. Mock external dependencies
5. Run tests: `pytest`

Example structure:

```python
class TestNewFeature:
    """Test description."""
    
    def test_specific_behavior(self, mock_vectorstore):
        """Test specific behavior."""
        # Arrange
        # Act
        # Assert
```

## Troubleshooting

### Tests require API keys
- All external APIs are mocked by default
- If tests fail with "No OPENAI_API_KEY", check conftest.py mocks

### ImportError when running tests
- Ensure you're in project directory
- Run: `pip install -e .` (if using setup.py)
- Or run tests from project root

### Mocking not working
- Check import paths in @patch decorators
- Use `@patch("module.Class")` where Class is used
- Verify mock is created before function call

## Performance

Test suite should complete in <10 seconds due to mocking:
- No network calls
- No file I/O (except temp fixtures)
- Lightweight document objects

For large-scale integration tests, mark as `@pytest.mark.slow` to skip during quick runs.

## Best Practices

1. **Test isolation**: Use fixtures to set up clean state
2. **Clear names**: Use descriptive test function names
3. **One assertion**: Focus each test on one behavior
4. **Mock external**: Mock all external dependencies
5. **Document**: Add docstrings explaining what's tested

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [LangChain testing guide](https://python.langchain.com/docs/guides/debugging)
