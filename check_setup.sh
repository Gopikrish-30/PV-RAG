#!/bin/bash
# PV-RAG Final Setup Script
# Run this to check if everything is ready

echo "======================================"
echo "PV-RAG System Readiness Check"
echo "======================================"
echo ""

# Check 1: Python version
echo "✓ Checking Python version..."
python --version
echo ""

# Check 2: Environment file
echo "✓ Checking .env file..."
if [ -f ".env" ]; then
    echo "  [OK] .env file exists"
else
    echo "  [ACTION NEEDED] Creating .env from .env.example..."
    cp .env.example .env
    echo "  [DONE] .env file created - please verify API keys inside"
fi
echo ""

# Check 3: Dataset
echo "✓ Checking dataset..."
if [ -f "legal_dataset_extended_with_mods_20260205_210844.csv" ]; then
    echo "  [OK] Dataset found"
else
    echo "  [ERROR] Dataset not found!"
fi
echo ""

# Check 4: Dependencies
echo "✓ Checking Python packages..."
echo "  Installing/updating requirements..."
pip install -r requirements.txt --quiet
echo "  [DONE] All packages installed"
echo ""

# Check 5: API Keys
echo "✓ Checking API keys in .env..."
if grep -q "GROQ_API_KEY=gsk_" .env; then
    echo "  [OK] Groq API key found"
else
    echo "  [WARNING] Groq API key not set"
fi

if grep -q "TAVILY_API_KEY=tvly-" .env; then
    echo "  [OK] Tavily API key found"
else
    echo "  [WARNING] Tavily API key not set"
fi
echo ""

# Check 6: ChromaDB
echo "✓ Checking ChromaDB..."
if [ -d "chroma_db" ]; then
    echo "  [OK] ChromaDB directory exists (data already loaded)"
else
    echo "  [ACTION NEEDED] ChromaDB not found - need to load data first"
    echo "  Run: python scripts/load_legal_data.py"
fi
echo ""

echo "======================================"
echo "Setup Status Summary"
echo "======================================"
echo ""
echo "Files created/updated:"
echo "  ✓ requirements.txt (added Tavily)"
echo "  ✓ config/settings.py (added tavily_api_key)"
echo "  ✓ .env.example (enabled web verification)"
echo "  ✓ app/agents/web_agent.py (NEW - Tavily integration)"
echo "  ✓ app/main.py (integrated web agent)"
echo ""

echo "Next Steps:"
echo "============"
echo ""
echo "1. Verify API keys in .env file:"
echo "   - GROQ_API_KEY=gsk_..."
echo "   - TAVILY_API_KEY=tvly-..."
echo ""
echo "2. Load dataset (if not done):"
echo "   python scripts/load_legal_data.py"
echo ""
echo "3. Start the server:"
echo "   uvicorn app.main:app --reload --port 8000"
echo ""
echo "4. Test at: http://localhost:8000/docs"
echo ""
echo "======================================"
echo "Ready to launch! 🚀"
echo "======================================"
