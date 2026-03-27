@echo off
REM RAG Test Runner Script for Windows
REM Usage: run_tests.bat [options]

setlocal enabledelayedexpansion

REM Default values
set VERBOSE=false
set COVERAGE=false

:parse_args
if "%~1"=="" goto :parse_done
if "%~1"=="-v" (set VERBOSE=true & shift & goto parse_args)
if "%~1"=="--verbose" (set VERBOSE=true & shift & goto parse_args)
if "%~1"=="-c" (set COVERAGE=true & shift & goto parse_args)
if "%~1"=="--coverage" (set COVERAGE=true & shift & goto parse_args)
if "%~1"=="-h" (goto help)
if "%~1"=="--help" (goto help)
echo Unknown option: %~1
exit /b 1

:parse_done
echo.
echo === RAG Unit Test Runner ===
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Install dependencies if needed
echo Checking dependencies...
pip install -q -r requirements.txt pytest pytest-mock 2>nul || pip install -q pytest pytest-mock

REM Run tests
echo.
echo ---
echo.

if "%VERBOSE%"=="true" (
    set PYTEST_ARGS=-v --tb=short
) else (
    set PYTEST_ARGS=-q --tb=line
)

if "%COVERAGE%"=="true" (
    echo Running tests with coverage...
    echo.
    pip install -q pytest-cov
    pytest %PYTEST_ARGS% --cov=. --cov-report=html --cov-report=term-missing
    echo.
    echo Coverage report generated: htmlcov\index.html
) else (
    echo Running tests...
    echo.
    pytest %PYTEST_ARGS%
)

echo.
echo ---
echo Tests completed
echo.
goto :end

:help
echo Usage: run_tests.bat [options]
echo Options:
echo   -v, --verbose     Show verbose output
echo   -c, --coverage    Generate coverage report
echo   -h, --help        Show this help message
exit /b 0

:end
endlocal
