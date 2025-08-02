@echo off
cd /d %~dp0
call rag-env\Scripts\activate.bat

echo Testing system components...

echo.
echo 1. Testing Python packages...
python -c "import sentence_transformers; print('? Sentence Transformers')"
python -c "import chromadb; print('? ChromaDB')"
python -c "import fitz; print('? PyMuPDF')"
python -c "import ollama; print('? Ollama client')"

echo.
echo 2. Testing Ollama connection...
ollama list

echo.
echo 3. Testing spec directory...
python src\download_specs.py

echo.
echo System test complete!
pause
