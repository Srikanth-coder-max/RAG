# Testing Instructions

This RAG application includes a comprehensive unit test suite to ensure code quality and reliability.

## Quick Start

### Run all tests
```bash
# Windows
run_tests.bat

# Linux/Mac
bash run_tests.sh
```

### Run with options
```bash
# Verbose output
run_tests.bat -v

# With coverage report
run_tests.bat -c

# Using pytest directly
pytest -v
pytest --cov=. --cov-report=html
```

## Test Suite Overview

The test suite includes **66 tests** covering:
- **Knowledge Base** (test_kb.py): PDF loading, splitting, vectorstore operations
- **RAG Core** (test_rag_core.py): LLM setup, embeddings, chain construction
- **Ingestion** (test_ingest.py): CLI interface and batch processing  
- **Streamlit App** (test_app.py): UI utilities and file operations

## Running Specific Tests

```bash
# Run single test file
pytest test_kb.py

# Run specific test class
pytest test_rag_core.py::TestGetLlm

# Run specific test
pytest test_app.py::TestAppUtilityFunctions::test_format_chat_history_empty

# Run tests matching pattern
pytest -k "embedding"

# Skip slow tests
pytest -m "not slow"
```

## Test Results Format

Typical output:
```
======================= 66 passed, 3 warnings in 27.11s =======================
```

All tests:
- ✅ Require no API keys (external APIs are mocked)
- ✅ Complete in < 30 seconds
- ✅ Work with no network access
- ✅ Are fully isolated from each other

## Coverage Reports

Generate an HTML coverage report:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html  # View in browser
```

## Dependencies

Testing requires:
- `pytest>=9.0.0`
- `pytest-mock>=3.15.0`

Install with:
```bash
pip install -r requirements.txt
pip install pytest pytest-mock
```

## Writing New Tests

1. Create test file: `test_module.py`
2. Import test utilities from conftest.py
3. Use existing fixtures or create new ones
4. Run: `pytest test_module.py -v`

Example test:
```python
def test_my_feature(mock_vectorstore, sample_documents):
    """Test the new feature."""
    # Arrange
    result = my_function(mock_vectorstore, sample_documents)
    
    # Assert
    assert result is not None
```

## CI/CD Integration

For GitHub Actions:
```yaml
- name: Run tests
  run: pytest --cov=. --cov-report= xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

**Tests fail with "No module named X"**
- Install dependencies: `pip install -r requirements.txt`
- Use virtual environment: `python -m venv .venv && source .venv/bin/activate`

**Tests are slow**
- Use `pytest -x` to stop on first failure
- Run specific tests instead of all: `pytest test_kb.py`

**Mock-related errors**
- Check @patch decorator paths match the import location
- Verify mock setup in conftest.py

## Additional Resources

See [TESTING.md](TESTING.md) for comprehensive testing guide.
