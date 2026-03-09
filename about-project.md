# PV-RAG: Proof-of-Validity Retrieval-Augmented Generation

**Project Name:** PV-RAG (Proof-of-Validity Retrieval-Augmented Generation for Legal Domain)

---

## 1. Introduction

PV-RAG represents a significant advancement in legal information retrieval by introducing **Temporal Validity Verification** an an essential component of the Retrieval-Augmented Generation (RAG) pipeline. 

Traditional RAG and legal search systems often fail in the legal domain because laws change frequently through amendments, multiple versions exist across different time periods, and users often need to know the historical state of a law ("What was valid *then*?") rather than just the current state. Furthermore, traditional systems do not automatically verify whether the retrieved information is outdated.

PV-RAG acts as a "time-traveling legal expert." It is an advanced AI system designed to provide **time-accurate** legal information by dynamically retrieving, validating, and presenting the correct version of Indian laws for any specified time period, backed by real-time web verification and provenance tracing.

---

## 2. Overview & Problem Statement

### The Problem
When dealing with legal inquiries (e.g., "What was the helmet fine in 2010?"):
1. **Google Search / Standard RAGs:** Often show current penalties (e.g., 2026 data), which is factually incorrect for the historical context.
2. **Static Legal Databases:** Might be years outdated or lack intuitive temporal queries.
3. **LLMs (ChatGPT/Claude):** Encode knowledge in frozen parameters, frequently hallucinating amendment dates, conflating different versions of a law, and offering zero source traceability.
4. **Current RAG architectures:** Retrieve documents based purely on semantic similarity, entirely ignoring time constraints and versioning.

### The PV-RAG Solution
PV-RAG solves this by introducing three highly novel concepts:
1. **Temporal Retrieval Layer:** Retrieving laws based not just on relevance, but on "validity at time T" using graph-based/version-chained metadata.
2. **Dual Verification System:** Using an offline structured temporal dataset for rapid historical queries, combined with an online Web Agent for real-time verification of current laws.
3. **Provenance-Backed Responses:** Every answer has a definitive source, a specific validity period, and an explicitly calculated confidence score.

---

## 3. About the Datasets

PV-RAG is powered by a massive, production-scale, temporally-versioned dataset built specifically for Indian Law.

### Dataset Specifications & Statistics
* **File Name:** `tlag_rag_dataset_full.csv` / `tlag_rag_dataset_full.parquet` (and historically `legal_dataset_extended_with_mods_20260205_210844.csv`)
* **Total Records:** 20,757 legal rules/sections.
* **Coverage:** 373 unique Central Acts.
* **Time Span:** 1860 - 2023 (163 years of legal history).
* **Versioning:** 5,155 rules (24.8%) have tracked version changes across 73 acts.
* **Temporal Tracking:** 100% of rows have a `start_year` and 92.8% have an `end_year`.
* **Completeness:** 100% contain the full legal `rule_text` and 91.7% contain official `source_url`s.

### Data Sources
The dataset is constructed by aggregating and extracting data from multiple highly credible sources:
1. **India Code** (`indiacode.nic.in`): Scraped for central acts and amendments.
2. **Gazette of India** (`egazette.nic.in`): For official amendment parsing.
3. **GitHub Repositories**: Structured JSON laws and constitutional data parsed from open-source legal repositories.
4. **Zenodo**: Academic legal datasets.
5. **Existing PV-RAG Base layer**.

### Core Features / Schema of the Dataset
Each row in the dataset represents **one version of one section in one time period**.
- `provision_id` & `version_id`: Unique identifiers for the rule and its specific time-version.
- `law_name` & `act_number`: The full title and number of the Act.
- `section`: Section identifier (e.g., Sec 129).
- `rule_text`: The complete legal text for that specific version.
- `start_year` / `start_date`: When this specific version enacted/became active.
- `end_year` / `end_date`: When this version was superseded or repealed (Null if still active).
- `penalty_fine_inr` & `penalty_imprisonment`: Extracted NLP features specifying exact penalty amounts.
- `status`: Active, superseded, or repealed.
- `source_url` & `source_label`: Provenance data linking directly back to India Code or Gazette.

---

## 4. Methodology & Core Features

### Core Modules
PV-RAG is orchestrated via FastAPI utilizing ChromaDB, Groq Qwen3-32B, and Tavily for advanced web search.

1. **Query Parsing Module:** An LLM-powered parser classifies the user query into types (`HISTORICAL_POINT`, `LATEST`, `DATE_RANGE`, `COMPARISON`, `GENERAL`), extracts temporal entities (years/dates), and rewrites the query for optimal vector search.
2. **Temporal Retrieval Engine:** A two-pass hybrid retrieval mechanism.
   * *Pass 1:* Pure Semantic Search (SentenceTransformer embeddings) for topic relevance.
   * *Pass 2:* Temporal-Filtered Search via ChromaDB metadata filters (`start_year <= query_year AND end_year >= query_year`).
   * *Merge:* Secondary matches receive a distance penalty/bonus to prioritize temporally-valid items over semantic-only matches.
3. **Validity Verification Layer (Web Agent):** An intelligent routing system. Historical queries skip web verification, while "latest" queries trigger a LangChain/Tavily web agent to search `gov.in`, `indiacode.nic.in`, etc., to confirm if a newer amendment exists beyond the dataset's cutoff.
4. **Context & Response Generator:** A synthesizer that takes the retrieved versions, the web agent's findings, and builds a comprehensive timeline and response.

### Key Innovations (Features)
* **Temporal Intelligence ⏰:** Accurately answers what was true in a specific year, maintaining a complete history of amendments.
* **Dual Verification 🔍:** The first RAG architecture combining offline structured database retrieval (fast, exact history) with live web verification (freshness guarantee).
* **Multi-Version Comparison 📈:** Generates a structured timeline of how a law changed (e.g., original vs. amended fines with percentage increases).
* **Provenance & Traceability 📚:** Dual-origin source attribution. Answers cite the exact Act, section, and India Code URL, plus the specific government web domain if live verification was used.
* **Dynamic Confidence Scoring 🎯:** Confidence scores automatically adjust based on source agreement (e.g., if the dataset and the web agent agree, confidence is >95%).

---

## 5. System Workflow

**Scenario: User asks: *"What is the current helmet fine?"***

1. **Query Understanding:** The parser detects `LATEST_STATUS`, extracts "current", and sets the target date to today. It flags the query as requiring Web Verification.
2. **Database Retrieval:** Using vector similarities, the system queries ChromaDB to find the most recent active version of "helmet fine" in the Motor Vehicles Act (e.g., v3: valid from 2019 to NULL). Dataset states ₹1,000.
3. **Web Agent Verification:** Because it's a "latest" query, the Web Agent searches Tavily using targeted domains (`gov.in`, `livelaw.in`) for recent amendments. 
4. **Consensus Analysis:** The LLM compares the dataset result (₹1,000) against the web result. If the web confirms no newer amendments exist since 2019, it returns `NO_UPDATE_FOUND`. 
5. **Response Generation:** The LLM outputs a final response: "The fine is ₹1,000." It includes the historical timeline (showing previous fines), attributes the sources (Dataset + Web), and assigns a high confidence score (~95%).

*(Note: For a historical query like "fine in 2010", the system skips the Web Agent, uses temporal SQL-style filters on the dataset to find the 2010 version, and instantly returns the historical 2010 penalty).*

---

## 6. Research Papers, Context, and Gaps

PV-RAG was developed after an extensive review of existing temporal RAG systems. Current research shows a massive gap in practical, legally-focused temporal RAG architectures.

### Existing Approaches & Their Flaws
* **TG-RAG / DyG-RAG:** Build complex temporal knowledge graphs (TKGs), mostly for the finance domain. **Gap:** They operate in an offline-only paradigm. They have no mechanism to validate the freshness of the retrieved data against real-world, live web parameters. 
* **SAT-Graph RAG & Deterministic Legal Agents:** Propose ontology-driven legal temporal reasoning. **Gap:** Purely conceptual blueprints; no code, no implementation, no quantitative datasets.
* **TimeR4:** Uses contrastive time-aware retriever fine-tuning. **Gap:** Depends on pre-existing TKGs and requires expensive fine-tuning.
* **ChatGPT/GPT-4 Base Models:** **Gap:** ChatGPT generates an answer interpolating from its training distribution. It physically cannot distinguish between `v1` and `v2` of a specific law, frequently hallucinating dates or providing current laws when asked about historical ones.

### PV-RAG's Unique Competitive Edge
PV-RAG is the **first** system to offer:
1. **Legal focus + Working Implementation + Web Verification + Production-Scale Dataset.**
2. A **Dual-Verification Architecture** offsetting the "knowledge cutoff" problem of offline datasets without losing the factual grounding required in law.
3. **Intelligent Routing** that saves compute by dynamically deciding whether an LLM web-check is required based on the temporal nature of the user's prompt.
4. **Structured "Open-Loop" Retrieval** ensuring that hallucinations regarding legal citations are structurally impossible because metrics (amounts, dates, acts) are explicitly mapped from `.csv`/`.parquet` metadata rather than parametrically generated.

---

## Conclusion
PV-RAG represents a paradigm shift. It transitions the application of AI in law from generating *plausible* legal text to retrieving computationally *verifiable*, temporally accurate legal facts. By ensuring every response is Time-Accurate, Source-Verified, Up-to-Date, and Transparent, PV-RAG serves as the optimal blueprint for future legal, compliance, and auditing AI systems.
