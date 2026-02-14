# PV-RAG Setup Guide

## Quick Start (Step-by-Step)

### Step 1: Install Python Dependencies

```bash
# Make sure you're in the PV-RAG directory
cd C:\Users\admin\Desktop\PV-RAG

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 2: Install PostgreSQL

**Option A: Using PostgreSQL Installer (Recommended for Windows)**
1. Download from: https://www.postgresql.org/download/windows/
2. Install PostgreSQL 14 or higher
3. During installation, remember your postgres password
4. Keep default port: 5432

**Option B: Using Docker**
```bash
docker run --name pvrag-postgres -e POSTGRES_PASSWORD=yourpassword -p 5432:5432 -d postgres:14
```

### Step 3: Create Database

Open Command Prompt or PostgreSQL pgAdmin and run:

```sql
CREATE DATABASE pvrag_db;
CREATE USER pvrag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE pvrag_db TO pvrag_user;
```

Or using psql command line:
```bash
psql -U postgres
# Then run the CREATE commands above
```

### Step 4: Configure Environment

Create a `.env` file in the project root:

```bash
# Copy from example
copy .env.example .env
```

Edit `.env` and update:
```ini
DATABASE_URL=postgresql://pvrag_user:your_password@localhost:5432/pvrag_db
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: Replace `your_password` and `your_openai_api_key_here` with actual values!

### Step 5: Setup Database

```bash
python scripts/setup_database.py
```

Expected output:
```
Creating database tables...
✓ All tables created successfully!
Created tables:
  - legal_rules
  - web_verification_logs
  - query_logs
```

### Step 6: Load Dataset

```bash
python scripts/load_legal_data.py
```

Expected output:
```
Loading dataset from: legal_dataset_extended_with_mods_20260205_210844.csv
Loaded 20757 records from CSV
...
✓ Data loading completed!
  ✓ Inserted: 20757
  - Skipped:  0
  ✗ Errors:   0
```

### Step 7: Run the Application

```bash
uvicorn app.main:app --reload --port 8000
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 8: Test the API

Open your browser and go to:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Testing with curl

### Test 1: Historical Query
```bash
curl -X POST "http://localhost:8000/api/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What was valid in 2010?\", \"enable_web_verification\": false}"
```

### Test 2: Latest Query
```bash
curl -X POST "http://localhost:8000/api/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is the current status?\", \"enable_web_verification\": false}"
```

### Test 3: Get Statistics
```bash
curl http://localhost:8000/api/stats
```

## Troubleshooting

### Issue: "No module named 'config'"
**Solution**: Make sure you're running from the project root directory

### Issue: "Connection refused" (Database)
**Solution**: 
1. Check PostgreSQL is running: `services.msc` → Find PostgreSQL
2. Verify connection string in `.env`
3. Test connection: `psql -U pvrag_user -d pvrag_db`

### Issue: "Table already exists"
**Solution**: Database was already setup. Skip setup_database.py

### Issue: Import errors
**Solution**: 
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: CSV file not found
**Solution**: Make sure `legal_dataset_extended_with_mods_20260205_210844.csv` is in the project root

## Project Structure

```
PV-RAG/
├── app/                          # Main application
│   ├── main.py                  # FastAPI app ✓
│   ├── modules/                 # Business logic
│   │   ├── query_parser.py     # Query understanding ✓
│   │   ├── retrieval.py        # Temporal retrieval ✓
│   │   ├── verification.py     # Web verification (TODO)
│   │   └── web_agent.py        # LangChain agent (TODO)
│   ├── models/                  # Database models
│   │   └── legal_models.py     # LegalRule, etc. ✓
│   └── db/                      # Database layer
│       └── session.py           # DB connection ✓
├── scripts/                     # Utility scripts
│   ├── setup_database.py       # DB initialization ✓
│   └── load_legal_data.py      # Data loader ✓
├── config/                      # Configuration
│   └── settings.py             # Settings ✓
├── requirements.txt            # Dependencies ✓
├── .env                        # Environment variables (create this!)
└── README.md                   # Documentation ✓
```

## Next Steps

1. ✓ Basic API is working
2. TODO: Implement Web Agent verification
3. TODO: Add vector database (ChromaDB) for semantic search
4. TODO: Integrate OpenAI/GPT-4 for response generation
5. TODO: Add proper logging
6. TODO: Write tests

## Development Workflow

```bash
# Activate environment
venv\Scripts\activate

# Run with auto-reload (for development)
uvicorn app.main:app --reload --port 8000

# Run tests (when implemented)
pytest tests/ -v

# Check logs
# Logs will appear in console
```

## Production Deployment (Future)

For production, you'll want to:
1. Use environment variables for secrets
2. Set DEBUG=False in .env
3. Use proper WSGI server (gunicorn on Linux)
4. Add authentication/authorization
5. Set up HTTPS
6. Configure CORS properly
7. Add monitoring and logging

## Questions?

Refer to:
- [Complete Overview](PV-RAG-Complete-Overview.txt)
- [Implementation Guide](PV-RAG-Implementation-Guide.md)
- API Docs: http://localhost:8000/docs (when running)
