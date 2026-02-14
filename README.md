# PV-RAG: Proof-of-Validity Retrieval-Augmented Generation

A temporal-aware legal question-answering system that retrieves time-accurate legal information with dual verification (Dataset + Web Agent).

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- **No PostgreSQL needed!** (Uses ChromaDB vector database)

### Installation

1. **Clone and Navigate**
```bash
cd PV-RAG
```

2. **Create Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
cp .env.example .env
# Optional: Add GROQ_API_KEY for LLM-powered responses (free!)
# Get key at: https://console.groq.com
```

5. **Load Dataset**
```bash
python scripts/load_legal_data.py
```

6. **Run Application**
```bash
uvicorn app.main:app --reload --port 8000
```

7. **Access API**
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

## 📁 Project Structure

```
PV-RAG/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/                    # API endpoints (future)
│   ├── modules/               # Core business logic
│   │   ├── query_parser.py   # Query understanding
│   │   ├── retrieval.py      # Temporal retrieval engine (ChromaDB)
│   │   └── response_gen.py   # Response generation (future)
│   ├── models/               # Data models
│   └── db/                   # Database connection
│       └── chromadb_manager.py  # ChromaDB vector store
├── scripts/                  # Utility scripts
│   └── load_legal_data.py   # CSV data loader → ChromaDB
├── config/                  # Configuration files
│   └── settings.py
├── tests/                   # Unit and integration tests
├── data/                    # Dataset directory
├── chroma_db/               # ChromaDB persistence (auto-created)
├── legal_dataset_extended_with_mods_20260205_210844.csv
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🎯 Key Features

- ⏰ **Temporal Intelligence**: Answers "What was valid in YYYY?"
- 🔍 **Vector Search**: Semantic search using ChromaDB embeddings
- 🤖 **Groq LLM**: Fast AI-powered responses (optional, free tier available!)
- 📈 **Multi-Version Timeline**: Complete amendment history
- 📚 **Source Attribution**: Every answer with provenance
- 🎯 **No Database Setup**: Everything in ChromaDB vector store
- ⚡ **Fast Setup**: Just Python + install + load data!

## 📊 Dataset

- **Records**: 20,757 legal rules/sections
- **Coverage**: 373 Central Acts (1860-2023)
- **Status**: ✅ Production Ready

## 🔧 API Usage

### Query Example
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the helmet fine in 2010?",
    "enable_web_verification": false
  }'
```

### Response
```json
{
  "answer": "As of 2010, the penalty was ₹500",
  "timeline": [
    {"period": "1999-2009", "value": "₹100", "status": "superseded"},
    {"period": "2009-2019", "value": "₹500", "status": "superseded"},
    {"period": "2019-Present", "value": "₹1,000", "status": "active"}
  ],
  "legal_reference": "Motor Vehicles Act, 1988 - Section 129",
  "confidence_score": 0.98,
  "verification_method": "dataset",
  "sources": ["Gazette of India, 2009"]
}
```

## 🤖 Groq LLM (Optional - Recommended!)

**Get professional AI-powered responses for FREE!** Takes 2 minutes:

### Quick Setup
1. **Get Free API Key**: https://console.groq.com
2. **Add to .env**: `GROQ_API_KEY=gsk_your_key_here`
3. **Restart server**: Done! ⚡

### Response Quality Comparison

**Without Groq (Template)**:
```
As of 2010, the relevant legal provision was under Income Tax Act, Section 80C.
```

**With Groq LLM**:
```
In 2010, Section 80C of the Income Tax Act, 1961 allowed deductions up to ₹1 lakh 
for investments in specified instruments. This provision was introduced in 2006 and 
remained effective until 2014 when the limit was increased to ₹1.5 lakh.
```

📖 **Full Guide**: [GROQ_SETUP.md](GROQ_SETUP.md)  
⚡ **Free Tier**: 14,400 requests/day, 300+ tokens/sec!

## 📖 Documentation

- [Complete Overview](PV-RAG-Complete-Overview.txt) - System methodology and workflow
- [Implementation Guide](PV-RAG-Implementation-Guide.md) - Technical details

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributors

PV-RAG Research Team

## 📧 Contact

For questions or contributions, see documentation files.
