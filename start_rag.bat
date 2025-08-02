@echo off
cd /d %~dp0
echo Starting Hardware RAG System...

rem Activate virtual environment
call rag-env\Scripts\activate.bat

rem Check if Ollama is running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting Ollama...
    start "" ollama serve
    timeout /t 3 >nul
)

rem Start RAG system
python src\rag_system.py

pause
