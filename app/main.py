"""
Main FastAPI Application (ChromaDB + Groq LLM + Tavily Web Agent)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import date, datetime
from pathlib import Path
from loguru import logger

from app.db.chromadb_manager import get_chroma_manager
from app.modules.query_parser import QueryParser
from app.modules.retrieval import TemporalRetrievalEngine
from app.modules.llm_response_generator import get_llm_generator, reset_llm_generator
from app.agents.web_agent import get_web_agent, reset_web_agent
from config.settings import settings, reload_settings


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

    Flow for HISTORICAL / GENERAL queries:
      1. LLM-powered query parsing (type, temporal, search_query)
      2. ChromaDB retrieval with temporal filtering
      3. Optional: Web verification & comparison via Tavily + LLM
      4. LLM answer generation
      5. Return structured response with timeline

    Flow for LATEST / CURRENT queries (web-only pipeline):
      1. LLM-powered query parsing → detects LATEST
      2. Multi-query rewriting (LLM generates 3-5 search variations)
      3. Web-only search across trusted government & legal sources
      4. LLM synthesises answer purely from web results (no dataset mixing)
      5. Return structured response with web sources
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

        # =============================================================
        # LATEST / CURRENT → Web-Only Pipeline (skip dataset entirely)
        # =============================================================
        if query_type == "LATEST":
            return await _handle_latest_web_only(request, parsed, start_time)

        # =============================================================
        # HISTORICAL / GENERAL / DATE_RANGE / COMPARISON → Dataset Pipeline
        # =============================================================
        return await _handle_dataset_pipeline(request, parsed, query_type, search_text, start_time)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_latest_web_only(
    request: QueryRequest,
    parsed: Dict,
    start_time: datetime,
) -> QueryResponse:
    """
    Web-only pipeline for LATEST/current queries.
    1. Multi-query rewriting
    2. Web search with all variations
    3. LLM answer from web results only (no dataset)
    """
    logger.info("LATEST query → routing to web-only pipeline (no dataset)")

    # Step 1: Multi-query rewriting
    multi_queries = query_parser.generate_multi_queries(request.question)
    logger.info(f"Multi-query rewriting: {len(multi_queries)} variations generated")

    # Step 2: Web-only search & answer
    web_result = web_agent.web_only_answer(
        query=request.question,
        multi_queries=multi_queries,
    )

    if web_result.get("answered") and web_result.get("answer"):
        answer = web_result["answer"]
        answer_source = "llm_web_only"
        confidence = web_result.get("confidence", 0.7)
        verification_method = "web_only (multi-query)"
    else:
        # Web failed — fall back to dataset as safety net
        logger.warning("Web-only pipeline returned no answer, falling back to dataset")
        retrieval = TemporalRetrievalEngine()
        search_text = parsed.get("search_query", request.question)
        rules = retrieval.retrieve_latest(
            query_text=search_text,
            limit=request.max_results,
        )
        if rules:
            answer = llm_generator.generate_response(
                query=request.question,
                query_type="LATEST",
                rules=rules,
                query_date=None,
                web_result=None,
            )
            answer_source = "llm_dataset_fallback"
            confidence = 0.6  # lower confidence since web failed
            verification_method = "dataset_fallback"
        else:
            answer = (
                "I couldn't find current legal information from web sources or the database. "
                "Please try asking with a specific Act name or Section number."
            )
            answer_source = "template"
            confidence = 0.0
            verification_method = "none"

    # Build sources
    sources: List[SourceInfo] = []
    if web_result.get("sources"):
        for web_url in web_result["sources"][:8]:
            domain = _extract_domain(web_url)
            # Label based on domain type
            if any(d in web_url for d in ["gov.in", "nic.in"]):
                label = f"Gov: {domain}"
            elif any(d in web_url for d in ["prsindia.org", "sansad.in"]):
                label = f"Official: {domain}"
            else:
                label = f"Web: {domain}"
            sources.append(SourceInfo(url=web_url, label=label, origin="web"))

    execution_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"LATEST query answered in {execution_time:.3f}s via web-only pipeline")

    return QueryResponse(
        answer=answer,
        query_type="LATEST",
        answer_source=answer_source,
        timeline=[],  # No timeline for web-only answers
        legal_reference="See web sources below",
        confidence_score=round(confidence, 3),
        verification_method=verification_method,
        sources=sources,
        query_date=date.today().isoformat(),
        execution_time_seconds=round(execution_time, 3),
    )


async def _handle_dataset_pipeline(
    request: QueryRequest,
    parsed: Dict,
    query_type: str,
    search_text: str,
    start_time: datetime,
) -> QueryResponse:
    """
    Dataset pipeline for HISTORICAL / GENERAL / DATE_RANGE / COMPARISON queries.
    """
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

    else:  # GENERAL, COMPARISON, …
        rules = retrieval.retrieve_general(
            query_text=search_text,
            limit=request.max_results,
        )

    if not rules:
        rules = retrieval.retrieve_general(
            query_text=request.question,
            limit=request.max_results,
        )

    if not rules:
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

    # ------ Step 3: Web verification (optional, for non-LATEST queries) ------
    web_result = None
    verification_method = "chromadb"

    should_verify = (
        request.enable_web_verification
        or settings.web_verification_enabled
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

    answer_source = getattr(llm_generator, '_last_source', 'template')

    base_conf = 0.92 if query_type in ("HISTORICAL_POINT",) else 0.85
    if web_result and web_result.get("verified"):
        base_conf = (base_conf + web_result.get("confidence", 0.5)) / 2

    sources: List[SourceInfo] = []
    law = primary.get("law_name", primary.get("act_title", "Unknown"))
    sec = primary.get("section", primary.get("section_id", "N/A"))

    src = primary.get("source", "")
    if src:
        sources.append(SourceInfo(url=src, label=f"Dataset: {law}", origin="dataset"))
    url = primary.get("source_url", "")
    if url:
        sources.append(SourceInfo(url=url, label=f"Dataset: {law} {sec}", origin="dataset"))
    if not sources:
        sources.append(SourceInfo(
            url="chromadb://legal_rules",
            label=f"Dataset: {law} – {sec}",
            origin="dataset",
        ))

    if web_result and web_result.get("sources"):
        for web_url in web_result["sources"][:5]:
            domain = _extract_domain(web_url)
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


def _extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "web"


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
    global query_parser, llm_generator, web_agent

    # Reload settings from .env to pick up any new API keys
    reload_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Vector DB: ChromaDB at {settings.chroma_persist_dir}")

    # Re-initialize all singletons with fresh settings (clears stale cached keys)
    query_parser = QueryParser()
    llm_generator = reset_llm_generator()
    web_agent = reset_web_agent()
    logger.info(f"API keys reloaded — Groq: {'set' if settings.groq_api_key else 'NOT set'}, "
                f"Tavily: {'set' if settings.tavily_api_key else 'NOT set'}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.app_name}")
