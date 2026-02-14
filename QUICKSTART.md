# PV-RAG Quick Start Guide ⚡

## Super Simple Setup (3 Steps!)

### ✅ Step 1: Install Python Dependencies
```bash
# Navigate to project
cd C:\Users\admin\Desktop\PV-RAG

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install packages (this will take a few minutes)
pip install -r requirements.txt
```

### ✅ Step 2: Load Your Dataset
```bash
# This loads your 20,757 legal rules into ChromaDB
python scripts/load_legal_data.py
```

Expected output:
```
Loading dataset from: legal_dataset_extended_with_mods_20260205_210844.csv
Loaded 20757 records from CSV
...
✓ Data loading completed!
  ✓ Inserted: 20757
```

### ✅ Step 3: Run the API!
```bash
uvicorn app.main:app --reload --port 8000
```

Visit: **http://localhost:8000/docs** to test your API! 🎉

---

## 🧪 Test Your API

### Test 1: Check Health
Open browser: http://localhost:8000/health

Should see:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-02-06T..."
}
```

### 🎯 **Optional: Enable Groq LLM for Better Responses**

Get **FREE** high-quality AI responses! Takes 2 minutes:

1. **Get Free API Key**: Visit https://console.groq.com
2. **Add to .env**:
   ```bash
   notepad .env
   # Add: GROQ_API_KEY=gsk_your_key_here
   ```
3. **Restart server**: Done! ⚡

Without Groq: Simple template responses (still works!)
With Groq: Professional, detailed legal explanations! 🚀

**Full Guide**: See [GROQ_SETUP.md](GROQ_SETUP.md)

---

### Test 2: Get Statistics
Open: http://localhost:8000/api/stats

Should see your 20,757 records!

### Test 3: Query Historical Data
In the API docs (http://localhost:8000/docs), try:

**POST /api/query**
```json
{
  "question": "What was valid in 2010?",
  "enable_web_verification": false,
  "max_results": 5
}
```

You should get a response with timeline and legal references!

---

## ❓ Troubleshooting

### ❌ Error: "No module named 'chromadb'"
**Fix**: Make sure you activated the virtual environment and installed requirements:
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### ❌ Error: "CSV file not found"
**Fix**: Make sure the CSV file is in the project root directory:
```bash
# Check if file exists
dir legal_dataset_extended_with_mods_20260205_210844.csv
```

### ❌ Error during data loading
**Fix**: Delete the ChromaDB folder and try again:
```bash
rmdir /s /q chroma_db
python scripts/load_legal_data.py
```

### ❌ API won't start
**Fix**: Check if port 8000 is already in use:
```bash
# Try a different port
uvicorn app.main:app --reload --port 8001
```

---

## 📁 What Just Happened?

1. **ChromaDB Created**: A `chroma_db/` folder was created with your vector database
2. **Embeddings Generated**: Each legal rule was converted to vector embeddings
3. **Metadata Stored**: All temporal data (start_year, end_year, etc.) stored as metadata
4. **API Ready**: FastAPI server ready to handle temporal queries!

---

## 🎯 What's Next?

- ✅ Your PV-RAG system is working!
- ✅ 20,757 legal rules indexed and searchable
- ✅ Temporal queries supported (historical + latest)
- ✅ Timeline generation enabled

**Optional Enhancements:**
- Add OpenAI API key to `.env` for better response generation
- Implement web verification for latest queries
- Add frontend UI (React/Next.js)
- Write tests

---

## 🚀 Key Advantages of ChromaDB Approach

✅ **No PostgreSQL Setup** - One less dependency!
✅ **Automatic Embeddings** - Semantic search built-in
✅ **Metadata Filtering** - Temporal queries still work perfectly
✅ **File-Based Storage** - Easy backup (just copy `chroma_db/` folder)
✅ **Fast Queries** - Vector similarity search is lightning fast
✅ **Simple Deployment** - Just package the `chroma_db/` folder

---

## 📚 Documentation

- **Complete Overview**: [PV-RAG-Complete-Overview.txt](PV-RAG-Complete-Overview.txt)
- **Implementation Guide**: [PV-RAG-Implementation-Guide.md](PV-RAG-Implementation-Guide.md)
- **API Documentation**: http://localhost:8000/docs (when running)

---

**You're all set!** 🎉 Your PV-RAG system is ready to answer temporal legal queries!
