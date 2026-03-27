#!/bin/bash
# RAG Test Runner Script
# Usage: bash run_tests.sh [options]

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
VERBOSE=false
COVERAGE=false
WATCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -w|--watch)
            WATCH=true
            shift
            ;;
        -h|--help)
            echo "Usage: bash run_tests.sh [options]"
            echo "Options:"
            echo "  -v, --verbose     Show verbose output"
            echo "  -c, --coverage    Generate coverage report"
            echo "  -w, --watch       Watch for changes and re-run tests"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== RAG Unit Test Runner ===${NC}\n"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python -m venv .venv
fi

# Activate virtual environment
source .venv/Scripts/activate

# Install dependencies if needed
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install -q -r requirements.txt pytest pytest-mock 2>/dev/null || pip install -q pytest pytest-mock

# Run tests
echo -e "${BLUE}---${NC}\n"

if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="-v --tb=short"
else
    PYTEST_ARGS="-q --tb=line"
fi

if [ "$COVERAGE" = true ]; then
    echo -e "${BLUE}Running tests with coverage...${NC}\n"
    pip install -q pytest-cov
    pytest ${PYTEST_ARGS} --cov=. --cov-report=html --cov-report=term-missing
    echo -e "\n${GREEN}Coverage report generated: htmlcov/index.html${NC}"
else
    echo -e "${BLUE}Running tests...${NC}\n"
    pytest ${PYTEST_ARGS}
fi

echo -e "\n${BLUE}---${NC}"
echo -e "${GREEN}✓ Tests completed${NC}\n"
