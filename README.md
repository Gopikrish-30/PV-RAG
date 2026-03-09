# PV-RAG: Proof-of-Validity Retrieval-Augmented Generation

**A Temporal-Aware Legal Question-Answering System for Indian Law**

PV-RAG is an advanced AI system that provides **time-accurate** legal information by dynamically retrieving, validating, and presenting the correct version of Indian laws for any specified time period — backed by real-time web verification and full provenance tracing.

> *"What was the helmet fine in 2010?"* — PV-RAG retrieves the exact provision valid in 2010, not the current one.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture Overview](#architecture-overview)
- [Key Features & Innovations](#key-features--innovations)
- [Dataset](#dataset)
- [System Workflow](#system-workflow)
- [Core Modules](#core-modules)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Tech Stack](#tech-stack)

---

## Problem Statement

When users ask temporal legal questions (e.g., *"What was the penalty for drunk driving in 2015?"*), existing systems fail:

| System | Problem |
|--------|---------|
| **Google Search** | Returns current penalties, factually incorrect for historical queries |
| **Static Legal Databases** | Often years outdated, lack intuitive temporal queries |
| **LLMs (ChatGPT/Claude)** | Hallucinate amendment dates, conflate versions, offer zero source traceability |
| **Standard RAG** | Retrieves by semantic similarity only, entirely ignoring time constraints and versioning |

**PV-RAG solves this** with three novel concepts:

1. **Temporal Retrieval Layer** — Retrieves laws based on *validity at time T*, not just relevance, using hybrid semantic+temporal search and ILO-Graph version chains.
2. **Dual Verification System** — Offline structured dataset for rapid historical queries + online Web Agent for real-time verification of current laws.
3. **Provenance-Backed Responses** — Every answer has a definitive source, validity period, and evidence-based confidence score.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                  │
│               "What was the helmet fine in 2010?"                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    MODULE 1: QUERY PARSER                            │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ LLM-Powered  │  │ Temporal Entity  │  │ Query Classification  │  │
│  │ Understanding │  │ Extraction       │  │ & Search Rewriting    │  │
│  └──────────────┘  └──────────────────┘  └───────────────────────┘  │
│  Output: type=HISTORICAL_POINT, year=2010, search="helmet fine MVA" │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     HISTORICAL/GENERAL          LATEST/CURRENT
              │                         │
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│ MODULE 2: TEMPORAL   │  │ MODULE 3: WEB AGENT (Tavily)             │
│ RETRIEVAL ENGINE     │  │                                          │
│                      │  │  Multi-Query Rewriting                   │
│ Pass 1: Semantic     │  │  → 3-5 search variations                 │
│    (embeddings)      │  │  Web Search (gov.in, indiacode.nic.in)   │
│ Pass 2: Temporal     │  │  → LLM synthesizes web-only answer       │
│    (year filter)     │  │                                          │
│ Merge + Boost        │  └──────────────────────────────────────────┘
│                      │
│ Version Chain        │
│ Enrichment           │
│ (all versions of     │
│  matched law+section)│
│                      │
│ ILO-Graph            │
│ Augmentation         │
│ (IPC→BNS successor   │
│  mapping)            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│              MODULE 4: RESPONSE GENERATOR (Groq LLM)                │
│                                                                      │
│  Retrieved Rules + Web Results + Timeline → Natural Language Answer  │
│  with citations, confidence score, and source attribution            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Features & Innovations

| Feature | Description |
|---------|-------------|
| **Temporal Intelligence** | Accurately answers *"What was valid in YYYY?"* — maintains complete amendment history across 163 years (1860–2023) |
| **Hybrid Retrieval** | Two-pass search: pure semantic (embedding similarity) + temporal-filtered (metadata year range), merged with temporal match boosting |
| **ILO-Graph Version Chains** | NetworkX knowledge graph (5,735 nodes, 9,440 edges) mapping act succession (IPC→BNS, CrPC→BNSS, IEA→BSA) and section-level version chains |
| **Dual Verification** | First RAG combining offline structured database (speed, exact history) with live web verification (freshness guarantee) |
| **Web-Only Pipeline** | LATEST/current queries bypass the dataset entirely — multi-query rewriting + Tavily web search + LLM synthesis from government sources |
| **Evidence-Based Confidence** | Dynamic scoring based on temporal validity, semantic distance, web agreement — not hardcoded values |
| **Provenance Tracing** | Every answer cites exact Act, Section, India Code URL, and web domain if live verification was used |
| **Version Timeline** | Structured chronological view of how a law changed over time, grouped by law+section |
| **Query Classification** | LLM-powered detection of 6 query types: HISTORICAL_POINT, LATEST, DATE_RANGE, COMPARISON, GENERAL, GREETING |
| **Comparison Queries** | Dedicated handler for *"Compare X in 2005 vs 2020"* — retrieves rules for both years with full version chains |

---

## Dataset

PV-RAG is powered by a production-scale, temporally-versioned dataset built specifically for Indian Law.

### Specifications

| Metric | Value |
|--------|-------|
| **Total Records** | 20,362 legal rules/sections (in ChromaDB) |
| **Coverage** | 373 unique Central Acts |
| **Time Span** | 1860–2023 (163 years of legal history) |
| **Versioned Rules** | ~24.8% have tracked version changes across 73 acts |
| **Temporal Tracking** | 100% have `start_year`, 92.8% have `end_year` |
| **Full Text** | 100% contain complete legal `rule_text` |
| **Source URLs** | 91.7% contain official India Code URLs |

### Data Sources

| Source | Type | URL |
|--------|------|-----|
| **India Code** | Central Acts & Amendments | `indiacode.nic.in` |
| **Gazette of India** | Official Amendment Notifications | `egazette.nic.in` |
| **GitHub Repositories** | Structured JSON laws (IPC, CrPC, Constitution) | Open-source legal repos |
| **Zenodo** | Academic legal datasets | `zenodo.org` |

### Schema

Each row represents **one version of one section in one time period**:

| Field | Description | Example |
|-------|-------------|---------|
| `rule_id` | Unique identifier | `MVA_194D_1988` |
| `law_name` | Full Act title | `THE MOTOR VEHICLES ACT, 1988` |
| `section` | Section identifier | `Sec194D` |
| `rule_text` | Complete legal text for this version | *"Whoever drives a motor cycle..."* |
| `start_year` | When this version became active | `1988` |
| `end_year` | When superseded (9999 = still active) | `1988` |
| `status` | Active, superseded, or repealed | `superseded` |
| `source` | Data source label | `AnnotatedCentralActs` |
| `source_url` | Official India Code URL | `https://indiacode.nic.in/...` |

### Data Collection Pipeline

The dataset is built by the scripts in `data_collection/`:

```
01_download_sources.py     → Download raw act data from India Code
02_scrape_indiacode.py     → Scrape structured section data
02b_extract_github_db.py   → Extract from GitHub legal repositories
03_scrape_gazette.py       → Parse Gazette of India amendments
04_enrich_versions.py      → Add temporal versioning metadata
05_build_dataset.py        → Build final unified CSV dataset
```

### ILO-Graph (Knowledge Graph)

Built by `scripts/build_graph.py`, the ILO-Graph captures:

| Metric | Value |
|--------|-------|
| **Nodes** | 5,735 (Legislation, Chapter, Section, LegalProvision nodes) |
| **Edges** | 9,440 (CONTAINS, HAS_VERSION, SUCCEEDS, REPEALS relationships) |
| **Legislation Acts** | 11 major acts including IPC, BNS, CrPC, BNSS, IEA, BSA |

**Act Succession Mapping:**
- Indian Penal Code (1860) → Bharatiya Nyaya Sanhita (2023)
- Criminal Procedure Code (1973) → Bharatiya Nagarik Suraksha Sanhita (2023)
- Indian Evidence Act (1872) → Bharatiya Sakshya Adhiniyam (2023)

**Section-Level Mapping (IPC→BNS examples):**

| IPC Section | BNS Section | Offense |
|-------------|-------------|---------|
| 302 | 103 | Murder |
| 304 | 105 | Culpable Homicide |
| 307 | 109 | Attempt to Murder |
| 376 | 64 | Rape |
| 420 | 318 | Cheating |
| 498A | 85 | Cruelty |

---

## System Workflow

### Historical Query Flow

**User asks:** *"What was the helmet fine in 2010?"*

```
1. QUERY PARSING
   → LLM detects: type=HISTORICAL_POINT, year=2010
   → Rewrites query: "helmet fine motor vehicles act penalty"

2. TEMPORAL RETRIEVAL (ChromaDB)
   → Pass 1: Semantic search finds MVA Sec194D (distance: 0.45)
   → Pass 2: Temporal filter (start_year ≤ 2010 AND end_year ≥ 2010)
   → Merge: temporal matches get distance boost (-0.15)

3. VERSION CHAIN ENRICHMENT
   → Fetches ALL versions of MVA Sec194D from ChromaDB
   → Builds complete amendment history for that section

4. GRAPH AUGMENTATION
   → Checks ILO-Graph for act succession (MVA has no successor → skip)
   → For IPC queries post-2024: maps IPC section → BNS equivalent

5. LLM RESPONSE GENERATION (Groq Qwen3-32B)
   → Input: retrieved rules + version chain + query context
   → Output: "The fine was Rs. 1,000 under Section 194D of MVA, 1988..."

6. CONFIDENCE SCORING
   → Base: 0.80 (HISTORICAL_POINT)
   → +0.10 (temporal validity match)
   → +0.04 (semantic distance < 1.0)
   → Final: 0.94
```

### Latest/Current Query Flow

**User asks:** *"What is the current punishment for murder?"*

```
1. QUERY PARSING
   → LLM detects: type=LATEST
   → Flags: needs web verification

2. MULTI-QUERY REWRITING
   → LLM generates 3-5 search variations:
     • "current punishment for murder India 2024"
     • "BNS Section 103 murder penalty"
     • "Bharatiya Nyaya Sanhita murder sentence"

3. WEB-ONLY PIPELINE (Tavily)
   → Searches gov.in, indiacode.nic.in, prsindia.org
   → Collects results from all query variations

4. LLM SYNTHESIS
   → Synthesizes answer purely from web results (no dataset mixing)
   → Cites actual government sources

5. RESPONSE
   → "Under BNS Section 103, murder is punishable with death or
      life imprisonment..."
   → Sources: indiacode.nic.in, prsindia.org
```

### Comparison Query Flow

**User asks:** *"Compare IPC Section 302 in 2005 vs 2020"*

```
1. QUERY PARSING → type=COMPARISON, year1=2005, year2=2020

2. DUAL RETRIEVAL
   → Retrieve rules valid at 2005-12-31
   → Retrieve rules valid at 2020-12-31

3. VERSION CHAIN ENRICHMENT
   → Fetch all versions for matched law+section pairs

4. LLM COMPARISON
   → Generates side-by-side analysis of changes

5. RESPONSE with timeline showing amendment progression
```

---

## Core Modules

### Module 1: Query Parser (`app/modules/query_parser.py`)

**Purpose:** Understand user intent using LLM + regex fallback.

| Feature | Description |
|---------|-------------|
| **Query Types** | HISTORICAL_POINT, LATEST, DATE_RANGE, COMPARISON, GENERAL, GREETING |
| **Temporal Extraction** | Detects years, date ranges, relative dates ("in 2010", "between 2005 and 2020") |
| **Search Rewriting** | Rewrites queries for optimal vector search relevance |
| **Multi-Query Rewriting** | Generates 3-5 search variations for broader web coverage |
| **Fallback** | Regex-based parsing when LLM is unavailable |

### Module 2: Temporal Retrieval Engine (`app/modules/retrieval.py`)

**Purpose:** Retrieve temporally-valid legal rules from ChromaDB + ILO-Graph.

| Method | Description |
|--------|-------------|
| `retrieve_with_graph()` | Full pipeline: semantic search → version chain enrichment → graph augmentation |
| `retrieve_for_point_in_time()` | Two-pass hybrid retrieval (semantic + temporal filter + merge) |
| `retrieve_for_comparison()` | Retrieves rules for two specific years with all version chains |
| `retrieve_latest()` | Hybrid retrieval boosting currently-active rules |
| `retrieve_by_date_range()` | Retrieves rules overlapping a year range |
| `build_timeline_dict()` | Groups results by law+section, deduplicates, sorts chronologically |
| `calculate_confidence()` | Evidence-based scoring from temporal validity, distance, web agreement |

**Hybrid Retrieval Strategy:**
```
Pass 1: Pure Semantic Search (SentenceTransformer embeddings)
         → Finds most relevant documents regardless of time
Pass 2: Temporal-Filtered Search (ChromaDB metadata filters)
         → start_year ≤ query_year AND end_year ≥ query_year
Merge:   Deduplicate by rule_id, boost temporal matches (L2 distance - 0.15)
```

### Module 3: Web Agent (`app/agents/web_agent.py`)

**Purpose:** Real-time web verification using Tavily search API.

| Feature | Description |
|---------|-------------|
| **Smart Routing** | `judge_need_for_web_verification()` decides if web search is needed |
| **verify_and_compare()** | Searches web, compares with dataset results, generates combined answer |
| **web_only_answer()** | For LATEST queries — synthesizes answer purely from web results |
| **Trusted Domains** | Prioritizes `gov.in`, `indiacode.nic.in`, `prsindia.org`, `sansad.in` |

### Module 4: LLM Response Generator (`app/modules/llm_response_generator.py`)

**Purpose:** Generate natural language answers using Groq LLM (Qwen3-32B).

| Feature | Description |
|---------|-------------|
| **Contextual Prompting** | Different prompts for HISTORICAL, LATEST, DATE_RANGE, COMPARISON queries |
| **Template Fallback** | Rich template responses when LLM is unavailable |
| **Think-Tag Handling** | Strips Qwen3's `<think>` reasoning tags, falls back if empty |
| **Source Tracking** | Tracks whether answer came from LLM+dataset, LLM+web, or template |

### Module 5: ChromaDB Manager (`app/db/chromadb_manager.py`)

**Purpose:** Vector database interface for semantic search with metadata filtering.

| Method | Description |
|--------|-------------|
| `semantic_search()` | Core embedding search with optional metadata filters |
| `query_by_point_in_time()` | Hybrid retrieval for a specific date |
| `query_latest()` | Hybrid retrieval boosting active rules |
| `query_by_date_range()` | Hybrid retrieval for year ranges |
| `query_timeline()` | Fetch all versions of a law+section (ChromaDB `get` with filters) |
| `_merge_results()` | Deduplicates and re-ranks two result sets with distance boosting |

### Module 6: Graph Manager (`app/db/graph_manager.py`)

**Purpose:** Interface to the ILO-Graph (NetworkX) knowledge graph.

| Method | Description |
|--------|-------------|
| `get_bns_equivalent()` | Map IPC section → BNS section number |
| `find_successor_act()` | Check if a law has been succeeded (IPC→BNS, CrPC→BNSS, IEA→BSA) |
| `get_version_at_year()` | Retrieve the specific version valid at a given year |
| `get_history_chain()` | Get full chain of versions with text and dates |
| `get_graph_stats()` | Return graph node/edge/legislation counts |

---

## Project Structure

```
PV-RAG/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application + all endpoints
│   ├── agents/
│   │   └── web_agent.py             # Tavily web search + verification agent
│   ├── db/
│   │   ├── chromadb_manager.py      # ChromaDB vector store manager
│   │   ├── graph_manager.py         # NetworkX ILO-Graph interface
│   │   ├── session.py               # Database session utilities
│   │   └── graph/
│   │       └── legal_graph.gpickle  # Pre-built ILO-Graph (5,735 nodes)
│   ├── models/
│   │   └── legal_models.py          # Pydantic data models
│   └── modules/
│       ├── query_parser.py          # LLM-powered query understanding
│       ├── retrieval.py             # Temporal retrieval engine (core)
│       └── llm_response_generator.py # Groq LLM response generation
│
├── config/
│   └── settings.py                  # Pydantic settings with .env support
│
├── scripts/
│   ├── load_legal_data.py           # CSV → ChromaDB data loader
│   ├── build_graph.py               # ILO-Graph builder (NetworkX)
│   ├── load_data_chromadb.py        # Alternative data loader
│   └── verify_retrieval_graph.py    # Graph retrieval verification
│
├── data_collection/
│   ├── scripts/                     # Data collection pipeline
│   │   ├── 01_download_sources.py   # Download raw act data
│   │   ├── 02_scrape_indiacode.py   # Scrape structured section data
│   │   ├── 02b_extract_github_db.py # Extract from GitHub repos
│   │   ├── 03_scrape_gazette.py     # Parse Gazette amendments
│   │   ├── 04_enrich_versions.py    # Add temporal versioning
│   │   └── 05_build_dataset.py      # Build final CSV dataset
│   ├── raw_acts/                    # Raw JSON act files
│   ├── gazette/                     # Gazette amendment data
│   ├── github_acts/                 # GitHub-sourced legal data
│   └── final_dataset/               # Processed output
│
├── static/
│   └── index.html                   # Chatbot UI (full web interface)
│
├── tests/
│   ├── test_pipeline.py             # End-to-end integration test
│   └── test_web_agent.py            # Web agent diagnostic test
│
├── chroma_db/                       # ChromaDB persistence (auto-created)
├── legal_dataset_*.csv              # Source dataset file
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
├── about-project.md                 # Detailed project description
├── GROQ_SETUP.md                    # Groq LLM setup guide
└── README.md                        # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+**
- No external databases required (ChromaDB is embedded)

### Step 1: Clone & Setup Environment

```bash
git clone <repository-url>
cd PV-RAG

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required for LLM-powered responses (free!)
GROQ_API_KEY=gsk_your_key_here
# Get free key at: https://console.groq.com

# Required for web verification (free tier available)
TAVILY_API_KEY=tvly-your_key_here
# Get key at: https://app.tavily.com
```

> **Without API keys:** The system still works — uses template-based responses and skips web verification. Adding Groq dramatically improves answer quality.

See [GROQ_SETUP.md](GROQ_SETUP.md) for detailed Groq setup instructions.

### Step 3: Load Dataset into ChromaDB

```bash
python scripts/load_legal_data.py
```

This reads the CSV dataset and loads all 20,362 legal rules into ChromaDB with embeddings.

### Step 4: Build ILO-Graph (Optional)

```bash
python scripts/build_graph.py
```

Builds the NetworkX knowledge graph from raw act data. A pre-built graph is included at `app/db/graph/legal_graph.gpickle`.

### Step 5: Run the Server

```bash
uvicorn app.main:app --port 8000 --host 0.0.0.0
```

### Step 6: Access the Application

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Chatbot UI (interactive web interface) |
| `http://localhost:8000/docs` | Swagger API documentation |
| `http://localhost:8000/redoc` | ReDoc API documentation |

---

## API Reference

### `POST /api/query` — Main Query Endpoint

**Request:**
```json
{
  "question": "What was the helmet fine in 2010?",
  "enable_web_verification": false,
  "max_results": 10
}
```

**Response:**
```json
{
  "answer": "In 2010, the fine for not wearing a helmet was Rs. 1,000 under Section 194D of the Motor Vehicles Act, 1988...",
  "query_type": "HISTORICAL_POINT",
  "answer_source": "llm_dataset",
  "timeline": [
    {
      "rule_id": "MVA_194D_1988",
      "period": "1988-1988",
      "start_date": "1988-01-01",
      "end_date": "1988-12-31",
      "status": "superseded",
      "version": null,
      "amendment_ref": null,
      "rule_text_preview": "THE MOTOR VEHICLES ACT, 1988 Sec194D Penalty for not wearing protective headgear..."
    }
  ],
  "legal_reference": "THE MOTOR VEHICLES ACT, 1988 - Section Sec194D",
  "confidence_score": 0.94,
  "verification_method": "chromadb",
  "sources": [
    {
      "url": "https://www.indiacode.nic.in/...",
      "label": "Dataset: THE MOTOR VEHICLES ACT, 1988 Sec194D",
      "origin": "dataset"
    }
  ],
  "query_date": "2010-12-31",
  "execution_time_seconds": 2.115
}
```

**Supported Query Types:**

| Query Type | Example | Pipeline |
|------------|---------|----------|
| `HISTORICAL_POINT` | "What was the helmet fine in 2010?" | ChromaDB + Graph + LLM |
| `LATEST` | "What is the current murder penalty?" | Web-only (Tavily + LLM) |
| `DATE_RANGE` | "Tax rules between 2005 and 2015" | ChromaDB + Version Chains |
| `COMPARISON` | "Compare IPC 302 in 2005 vs 2020" | Dual retrieval + LLM comparison |
| `GENERAL` | "Tell me about the Motor Vehicles Act" | Pure semantic search + LLM |
| `GREETING` | "Hello" | Template response |

### `POST /api/timeline` — Version Timeline

Returns the complete amendment history for a specific law and section.

**Request:**
```json
{
  "law_name": "THE MOTOR VEHICLES ACT, 1988",
  "section": "Sec194D"
}
```

**Response:**
```json
{
  "law_name": "THE MOTOR VEHICLES ACT, 1988",
  "section": "Sec194D",
  "version_count": 1,
  "timeline": [
    {
      "rule_id": "MVA_194D_1988",
      "period": "1988-1988",
      "status": "superseded",
      "version": "v1",
      "rule_text_preview": "..."
    }
  ]
}
```

### `GET /api/stats` — System Statistics

**Response:**
```json
{
  "total_rules": 20362,
  "unique_acts_sample": 21,
  "active_rules_sample": 0,
  "storage": "ChromaDB (Vector Database)",
  "graph": {
    "available": true,
    "nodes": 5735,
    "edges": 9440,
    "legislation_count": 11
  }
}
```

### `GET /health` — Health Check

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-03-08T22:54:35.363888",
  "document_count": 20362
}
```

---

## Configuration

All settings are managed via environment variables (`.env` file) with sensible defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key for LLM responses |
| `GROQ_MODEL` | `qwen/qwen3-32b` | Groq model name |
| `TAVILY_API_KEY` | — | Tavily API key for web search |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB data directory |
| `CHROMA_COLLECTION_NAME` | `legal_rules` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `WEB_VERIFICATION_ENABLED` | `False` | Enable web verification for all queries |
| `DATASET_PATH` | `./legal_dataset_*.csv` | Path to source CSV dataset |
| `DATASET_CUTOFF_YEAR` | `2020` | Dataset freshness cutoff |
| `MIN_CONFIDENCE_SCORE` | `0.7` | Minimum confidence threshold |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Test the full pipeline (ChromaDB → retrieval → LLM)
python tests/test_pipeline.py

# Test the web agent
python tests/test_web_agent.py
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | REST API with async support |
| **Vector Database** | ChromaDB (PersistentClient) | Semantic search with metadata filtering |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Document and query embeddings |
| **Knowledge Graph** | NetworkX (MultiDiGraph) | ILO-Graph for version chains and act succession |
| **LLM** | Groq (`qwen/qwen3-32b`) | Query parsing, response generation, web synthesis |
| **Web Search** | Tavily | Real-time legal information verification |
| **LLM Framework** | LangChain | LLM integration and prompt management |
| **Frontend** | HTML/CSS/JS | Chatbot UI served as static files |
| **Configuration** | Pydantic Settings + dotenv | Type-safe settings with .env support |
| **Logging** | Loguru | Structured logging throughout |

---

## License

MIT License — See LICENSE file for details.

## Contributors

PV-RAG Research Team

## Links

- [Detailed Project Description](about-project.md)
- [Groq LLM Setup Guide](GROQ_SETUP.md)
- API Documentation: `http://localhost:8000/docs` (when server is running)
