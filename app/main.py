"""
Main FastAPI Application (ChromaDB + Groq LLM + Tavily Web Agent)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, datetime
from pathlib import Path
from loguru import logger

from app.db.chromadb_manager import get_chroma_manager
from app.modules.query_parser import QueryParser
from app.modules.retrieval import TemporalRetrievalEngine
from app.modules.llm_response_generator import get_llm_generator
from app.agents.web_agent import get_web_agent
from config.settings import settings


# =====================================================================
# FastAPI setup
# =====================================================================
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Proof-of-Validity Retrieval-Augmented Generation for Legal Domain",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons – initialised once at import time
query_parser = QueryParser()
llm_generator = get_llm_generator()
web_agent = get_web_agent()

# Static files for chatbot UI
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =====================================================================
# Request / Response models
# =====================================================================

class QueryRequest(BaseModel):
    question: str
    enable_web_verification: bool = False
    max_results: int = 10


class TimelineEntry(BaseModel):
    rule_id: str
    period: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status: str
    version: Optional[str] = None
    amendment_ref: Optional[str] = None
    rule_text_preview: str


class SourceInfo(BaseModel):
    """Individual source with origin label"""
    url: str
    label: str          # e.g. "Dataset", "Web (gov.in)", "Web (livelaw.in)"
    origin: str         # "dataset" or "web"


class QueryResponse(BaseModel):
    answer: str
    query_type: str
    answer_source: str                # "llm_dataset", "llm_web_merged", "llm_dataset+web", "template"
    timeline: List[TimelineEntry]
    legal_reference: str
    confidence_score: float
    verification_method: str
    sources: List[SourceInfo]
    query_date: Optional[str] = None
    execution_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    document_count: int = 0


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the chatbot UI."""
    html = Path(__file__).parent.parent / "static" / "index.html"
    if html.exists():
        return FileResponse(html)
    return {"message": f"Welcome to {settings.app_name}", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        chroma = get_chroma_manager()
        stats = chroma.get_stats()
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            timestamp=datetime.now().isoformat(),
            document_count=stats["total_rules"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/api/query", response_model=QueryResponse)
async def query_legal_information(request: QueryRequest):
    """
    Main query endpoint.

    Flow:
      1. LLM-powered query parsing (type, temporal, search_query)
      2. ChromaDB retrieval with temporal filtering
      3. Optional: Web verification & comparison via Tavily + LLM
      4. LLM answer generation
      5. Return structured response with timeline
    """
    start_time = datetime.now()

    try:
        logger.info(f"Processing query: {request.question}")

        # ------ Step 1: Parse query ------
        parsed = query_parser.parse(request.question)
        query_type = parsed["query_type"]
        search_text = parsed.get("search_query", request.question)
        logger.info(f"Parsed → type={query_type}, search_text='{search_text[:60]}'")

        # Handle greetings / off-topic
        if query_type == "GREETING":
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResponse(
                answer=(
                    "Hello! I'm PV-RAG, your Indian Legal Information Assistant. \n\n"
                    "You can ask me questions like:\n"
                    "- *What is the penalty for not wearing a helmet?*\n"
                    "- *What was the income tax rate in 2010?*\n"
                    "- *Current drunk driving penalty under Motor Vehicles Act*\n\n"
                    "How can I help you today?"
                ),
                query_type="GREETING",
                answer_source="template",
                timeline=[],
                legal_reference="N/A",
                confidence_score=1.0,
                verification_method="none",
                sources=[],
                query_date=None,
                execution_time_seconds=round(execution_time, 3),
            )

        # ------ Step 2: Retrieve from ChromaDB ------
        retrieval = TemporalRetrievalEngine()
        rules: list = []
        query_date_str: Optional[str] = None

        if query_type == "HISTORICAL_POINT":
            qd = parsed["temporal_entities"].get("date")
            query_date_str = qd.isoformat() if qd else None
            rules = retrieval.retrieve_for_point_in_time(
                query_text=search_text,
                query_date=qd or date.today(),
                limit=request.max_results,
            )

        elif query_type == "DATE_RANGE":
            sy = parsed["temporal_entities"].get("start_year", 1900)
            ey = parsed["temporal_entities"].get("end_year", date.today().year)
            rules = retrieval.retrieve_by_date_range(
                query_text=search_text,
                start_year=sy,
                end_year=ey,
                limit=request.max_results,
            )

        elif query_type == "LATEST":
            query_date_str = date.today().isoformat()
            rules = retrieval.retrieve_latest(
                query_text=search_text,
                limit=request.max_results,
            )

        else:  # GENERAL, COMPARISON, …
            rules = retrieval.retrieve_general(
                query_text=search_text,
                limit=request.max_results,
            )

        if not rules:
            # Last resort: pure semantic search with the original question
            rules = retrieval.retrieve_general(
                query_text=request.question,
                limit=request.max_results,
            )

        if not rules:
            # Still nothing – return a helpful "no results" answer instead of crashing
            execution_time = (datetime.now() - start_time).total_seconds()
            return QueryResponse(
                answer=(
                    "I couldn't find any matching legal rules in the database for your query. "
                    "Please try rephrasing your question or specifying the act name, section, "
                    "or topic you're interested in."
                ),
                query_type=query_type,
                answer_source="template",
                timeline=[],
                legal_reference="N/A",
                confidence_score=0.0,
                verification_method="none",
                sources=[],
                query_date=query_date_str,
                execution_time_seconds=round(execution_time, 3),
            )

        # ------ Step 3: Web verification (optional) ------
        web_result = None
        verification_method = "chromadb"

        # Auto-enable web verification for LATEST queries (user wants
        # current info compared with web results)
        should_verify = (
            request.enable_web_verification
            or settings.web_verification_enabled
            or query_type == "LATEST"
            or parsed.get("needs_web_verification", False)
        )
        if should_verify:
            need_web = web_agent.judge_need_for_web_verification(
                query=request.question,
                query_type=query_type,
                retrieved_rules=rules,
                query_date=parsed["temporal_entities"].get("date"),
            )
            if need_web:
                logger.info("Performing web verification…")
                web_result = web_agent.verify_and_compare(
                    query=request.question,
                    dataset_rules=rules,
                )
                if web_result and web_result.get("verified"):
                    verification_method = "chromadb + web"
                    logger.info(f"Web verified (confidence {web_result.get('confidence', 0):.2f})")

        # ------ Step 4: Generate answer ------
        query_date_obj = parsed["temporal_entities"].get("date") if query_type == "HISTORICAL_POINT" else None
        answer = llm_generator.generate_response(
            query=request.question,
            query_type=query_type,
            rules=rules,
            query_date=query_date_obj,
            web_result=web_result,
        )

        # ------ Step 5: Build response ------
        primary = rules[0]
        timeline = retrieval.build_timeline_dict(rules)

        # Answer source tracking
        answer_source = getattr(llm_generator, '_last_source', 'template')

        # Confidence
        base_conf = 0.92 if query_type in ("HISTORICAL_POINT", "LATEST") else 0.85
        if web_result and web_result.get("verified"):
            base_conf = (base_conf + web_result.get("confidence", 0.5)) / 2

        # Sources — structured with origin labels
        sources: List[SourceInfo] = []
        law = primary.get("law_name", primary.get("act_title", "Unknown"))
        sec = primary.get("section", primary.get("section_id", "N/A"))

        # Dataset sources
        src = primary.get("source", "")
        if src:
            sources.append(SourceInfo(url=src, label=f"Dataset: {law}", origin="dataset"))
        url = primary.get("source_url", "")
        if url:
            sources.append(SourceInfo(url=url, label=f"Dataset: {law} {sec}", origin="dataset"))
        # If no source/url in metadata, still credit the dataset
        if not sources:
            sources.append(SourceInfo(
                url="chromadb://legal_rules",
                label=f"Dataset: {law} – {sec}",
                origin="dataset",
            ))

        # Web sources (with domain labels)
        if web_result and web_result.get("sources"):
            for web_url in web_result["sources"][:5]:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(web_url).netloc.replace("www.", "")
                except Exception:
                    domain = "web"
                sources.append(SourceInfo(
                    url=web_url,
                    label=f"Web: {domain}",
                    origin="web",
                ))

        execution_time = (datetime.now() - start_time).total_seconds()

        response = QueryResponse(
            answer=answer,
            query_type=query_type,
            answer_source=answer_source,
            timeline=[TimelineEntry(**e) for e in timeline],
            legal_reference=f"{law} – Section {sec}",
            confidence_score=round(base_conf, 3),
            verification_method=verification_method,
            sources=sources,
            query_date=query_date_str,
            execution_time_seconds=round(execution_time, 3),
        )

        logger.info(f"Query answered in {execution_time:.3f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    try:
        chroma = get_chroma_manager()
        stats = chroma.get_stats()
        return {
            "total_rules": stats["total_rules"],
            "unique_acts_sample": stats.get("unique_acts_sample", 0),
            "active_rules_sample": stats.get("active_rules_sample", 0),
            "storage": "ChromaDB (Vector Database)",
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# Lifecycle
# =====================================================================

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Vector DB: ChromaDB at {settings.chroma_persist_dir}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")
