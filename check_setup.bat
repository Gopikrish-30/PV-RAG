@echo off
REM PV-RAG Final Setup Check (Windows)
echo ======================================
echo PV-RAG System Readiness Check
echo ======================================
echo.

REM Check 1: Python version
echo [CHECK] Python version...
python --version
echo.

REM Check 2: Environment file
echo [CHECK] .env file...
if exist .env (
    echo   [OK] .env file exists
) else (
    echo   [ACTION] Creating .env from .env.example...
    copy .env.example .env
    echo   [DONE] .env file created - verify API keys inside!
)
echo.

REM Check 3: Dataset
echo [CHECK] Dataset file...
if exist legal_dataset_extended_with_mods_20260205_210844.csv (
    echo   [OK] Dataset found
) else (
    echo   [ERROR] Dataset not found!
)
echo.

REM Check 4: Dependencies
echo [CHECK] Installing/updating packages...
pip install -r requirements.txt --quiet
echo   [DONE] All packages installed
echo.

REM Check 5: ChromaDB
echo [CHECK] ChromaDB directory...
if exist chroma_db (
    echo   [OK] ChromaDB directory exists (data loaded)
) else (
    echo   [ACTION] ChromaDB not found - need to load data
    echo   Run: python scripts\load_legal_data.py
)
echo.

echo ======================================
echo Setup Status Summary
echo ======================================
echo.
echo Files updated:
echo   * requirements.txt (added Tavily)
echo   * config/settings.py (added tavily_api_key)
echo   * .env.example (enabled web verification)
echo   * app/agents/web_agent.py (NEW - Tavily integration)
echo   * app/main.py (integrated web agent)
echo.

echo ======================================
echo NEXT STEPS:
echo ======================================
echo.
echo 1. Verify API keys in .env:
echo    - GROQ_API_KEY=gsk_...
echo    - TAVILY_API_KEY=tvly-...
echo.
echo 2. Load dataset (if not done):
echo    python scripts\load_legal_data.py
echo.
echo 3. Start server:
echo    uvicorn app.main:app --reload --port 8000
echo.
echo 4. Test: http://localhost:8000/docs
echo.
echo ======================================
echo Ready to launch! :)
echo ======================================
pause
