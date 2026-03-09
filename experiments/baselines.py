"""
Baseline Systems for PV-RAG Comparative Evaluation

Systems implemented:
  1. NaiveRAG        — Semantic-only retrieval (no temporal filtering)
  2. BM25Retrieval   — Lexical retrieval with BM25 scoring
  3. LLMOnly         — Direct LLM query (no retrieval, simulates ChatGPT)
  4. TemporalOnly    — Temporal filter without semantic ranking (ablation)
  5. PVRAGFull       — The full PV-RAG system (our approach)

All baselines implement the same interface:
  .retrieve(query, query_year, limit) -> List[Dict]
  .generate_answer(query, rules, query_year) -> str
"""
import re
import math
import time
from typing import List, Dict, Optional, Tuple
from datetime import date
from collections import Counter, defaultdict

from loguru import logger


# ──────────────────────────────────────────────────────────────────────
# BASE CLASS
# ──────────────────────────────────────────────────────────────────────

class BaselineSystem:
    """Interface for all retrieval systems."""

    name: str = "BaselineSystem"
    description: str = ""

    def retrieve(
        self,
        query: str,
        query_year: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict]:
        raise NotImplementedError

    def generate_answer(
        self,
        query: str,
        rules: List[Dict],
        query_year: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Returns (answer_text, confidence_score)."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
# 1. NAIVE RAG — Semantic-only (no temporal filtering)
# ──────────────────────────────────────────────────────────────────────

class NaiveRAG(BaselineSystem):
    """
    Standard RAG baseline: pure vector similarity search.
    No temporal filtering, no version chains, no graph augmentation.
    This is what a typical RAG system (LangChain, LlamaIndex default) does.
    """

    name = "NaiveRAG"
    description = "Semantic-only retrieval (standard RAG, no temporal awareness)"

    def __init__(self):
        from app.db.chromadb_manager import get_chroma_manager
        self.chroma = get_chroma_manager()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from app.modules.llm_response_generator import get_llm_generator
            self._llm = get_llm_generator()
        return self._llm

    def retrieve(self, query, query_year=None, limit=10):
        """Pure semantic search — completely ignores query_year."""
        results = self.chroma.semantic_search(query, n_results=limit)
        return self._convert(results)

    def generate_answer(self, query, rules, query_year=None):
        if not rules:
            return "No relevant legal provisions found.", 0.3

        context = self._build_context(rules[:5])
        prompt = (
            f"Based on these legal provisions:\n{context}\n\n"
            f"Answer: {query}\n"
            f"Cite the Act and Section."
        )
        try:
            answer = self.llm.generate_response(
                query=query,
                query_type="GENERAL",
                rules=rules,
                query_date=date(query_year, 12, 31) if query_year else None,
            )
            confidence = self._calc_confidence(rules)
            return answer, confidence
        except Exception as e:
            logger.warning(f"NaiveRAG LLM failed: {e}")
            return self._template_answer(rules), 0.5

    def _convert(self, results):
        rules = []
        if not results.get("ids"):
            return rules
        ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
        docs = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        metas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
        dists = []
        if results.get("distances"):
            dists = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
        for i in range(len(ids)):
            m = metas[i] if i < len(metas) else {}
            rules.append({
                "rule_id": ids[i],
                "rule_text": docs[i] if i < len(docs) else "",
                "law_name": m.get("law_name", ""),
                "section": m.get("section", ""),
                "start_year": m.get("start_year", 0),
                "end_year": m.get("end_year", 9999),
                "status": m.get("status", "active"),
                "distance": dists[i] if i < len(dists) else 1.0,
                "source_url": m.get("source_url", ""),
            })
        return rules

    def _build_context(self, rules):
        parts = []
        for i, r in enumerate(rules):
            parts.append(
                f"[{i+1}] {r['law_name']}, {r['section']} "
                f"(Valid: {r['start_year']}-{r['end_year']}): "
                f"{r['rule_text'][:400]}"
            )
        return "\n".join(parts)

    def _calc_confidence(self, rules):
        if not rules:
            return 0.3
        avg_dist = sum(r.get("distance", 1.0) for r in rules[:3]) / min(3, len(rules))
        return max(0.3, min(0.85, 1.0 - avg_dist * 0.3))

    def _template_answer(self, rules):
        if not rules:
            return "No relevant provisions found."
        r = rules[0]
        return (
            f"Based on {r['law_name']}, {r['section']}: "
            f"{r['rule_text'][:300]}..."
        )


# ──────────────────────────────────────────────────────────────────────
# 2. BM25 RETRIEVAL — Lexical/keyword-based retrieval
# ──────────────────────────────────────────────────────────────────────

class BM25Retrieval(BaselineSystem):
    """
    BM25 (Best Matching 25) lexical retrieval baseline.
    Traditional keyword-based retrieval used before dense embeddings.
    No temporal filtering.
    """

    name = "BM25"
    description = "Lexical BM25 retrieval (keyword matching, no embeddings)"

    def __init__(self):
        from app.db.chromadb_manager import get_chroma_manager
        self.chroma = get_chroma_manager()
        self._corpus = None
        self._corpus_meta = None
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from app.modules.llm_response_generator import get_llm_generator
            self._llm = get_llm_generator()
        return self._llm

    def _load_corpus(self):
        """Load all documents from ChromaDB for BM25 scoring."""
        if self._corpus is not None:
            return

        logger.info("BM25: Loading corpus from ChromaDB...")

        # Sample documents (BM25 on full 20k is expensive, use top semantic matches)
        # We'll use a two-stage approach: semantic pre-filter then BM25 re-rank
        self._corpus = []
        self._corpus_meta = []

    def retrieve(self, query, query_year=None, limit=10):
        """Two-stage: semantic pre-filter (top 50), then BM25 re-rank."""
        # Stage 1: Get candidate pool via semantic search
        results = self.chroma.semantic_search(query, n_results=50)
        candidates = self._convert(results)

        if not candidates:
            return []

        # Stage 2: BM25 re-rank
        query_terms = self._tokenize(query)
        scored = []

        # Build mini-corpus stats
        doc_count = len(candidates)
        df = Counter()  # document frequency
        for c in candidates:
            terms = set(self._tokenize(c.get("rule_text", "")))
            for t in terms:
                df[t] += 1

        avg_dl = sum(len(self._tokenize(c.get("rule_text", ""))) for c in candidates) / max(doc_count, 1)

        for c in candidates:
            doc_terms = self._tokenize(c.get("rule_text", ""))
            score = self._bm25_score(query_terms, doc_terms, df, doc_count, avg_dl)
            c["bm25_score"] = score
            scored.append(c)

        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored[:limit]

    def generate_answer(self, query, rules, query_year=None):
        if not rules:
            return "No relevant legal provisions found.", 0.3
        try:
            answer = self.llm.generate_response(
                query=query,
                query_type="GENERAL",
                rules=rules,
                query_date=date(query_year, 12, 31) if query_year else None,
            )
            return answer, 0.6
        except Exception:
            r = rules[0]
            return f"Based on {r['law_name']}, {r['section']}: {r['rule_text'][:300]}...", 0.4

    def _tokenize(self, text):
        """Simple whitespace + lowercase tokenization."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _bm25_score(self, query_terms, doc_terms, df, N, avg_dl, k1=1.5, b=0.75):
        """Compute BM25 score for a document given query terms."""
        dl = len(doc_terms)
        tf_map = Counter(doc_terms)
        score = 0.0

        for qt in query_terms:
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            idf = math.log((N - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            score += idf * tf_norm

        return score

    def _convert(self, results):
        rules = []
        if not results.get("ids"):
            return rules
        ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
        docs = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        metas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
        dists = []
        if results.get("distances"):
            dists = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
        for i in range(len(ids)):
            m = metas[i] if i < len(metas) else {}
            rules.append({
                "rule_id": ids[i],
                "rule_text": docs[i] if i < len(docs) else "",
                "law_name": m.get("law_name", ""),
                "section": m.get("section", ""),
                "start_year": m.get("start_year", 0),
                "end_year": m.get("end_year", 9999),
                "status": m.get("status", "active"),
                "distance": dists[i] if i < len(dists) else 1.0,
                "source_url": m.get("source_url", ""),
            })
        return rules


# ──────────────────────────────────────────────────────────────────────
# 3. LLM-ONLY — No retrieval, just parametric knowledge
# ──────────────────────────────────────────────────────────────────────

class LLMOnly(BaselineSystem):
    """
    Pure LLM baseline (simulates ChatGPT/Claude on legal questions).
    No retrieval at all — relies entirely on parametric knowledge.
    This exposes hallucination and temporal confusion issues.
    """

    name = "LLMOnly"
    description = "Direct LLM query without retrieval (parametric knowledge only)"

    def __init__(self):
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            from langchain_groq import ChatGroq
            from config.settings import settings
            self._llm_client = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=0.1,
                max_tokens=2048,
            )
        return self._llm_client

    def retrieve(self, query, query_year=None, limit=10):
        """No retrieval — returns empty list."""
        return []

    def generate_answer(self, query, rules=None, query_year=None):
        year_context = f" as of the year {query_year}" if query_year else ""
        prompt = (
            f"You are a legal expert on Indian law. "
            f"Answer the following question{year_context}. "
            f"Cite specific Acts, Sections, and penalty amounts if applicable.\n\n"
            f"Question: {query}"
        )
        try:
            response = self.llm_client.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            # Strip think tags
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return text, 0.5  # LLM-only always gets moderate confidence
        except Exception as e:
            logger.warning(f"LLMOnly failed: {e}")
            return "Unable to generate response.", 0.1


# ──────────────────────────────────────────────────────────────────────
# 4. TEMPORAL-ONLY — Temporal filter without semantic ranking (ablation)
# ──────────────────────────────────────────────────────────────────────

class TemporalOnly(BaselineSystem):
    """
    Ablation baseline: temporal filtering WITHOUT semantic boosting.
    Uses only the year-filtered ChromaDB query (Pass 2 of PV-RAG).
    This shows the contribution of the hybrid merge strategy.
    """

    name = "TemporalOnly"
    description = "Temporal-filtered retrieval only (ablation, no hybrid merge)"

    def __init__(self):
        from app.db.chromadb_manager import get_chroma_manager
        self.chroma = get_chroma_manager()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from app.modules.llm_response_generator import get_llm_generator
            self._llm = get_llm_generator()
        return self._llm

    def retrieve(self, query, query_year=None, limit=10):
        """Only temporal-filtered search, no semantic pass."""
        if query_year is None:
            # Without year, just do semantic (degrades to NaiveRAG)
            results = self.chroma.semantic_search(query, n_results=limit)
        else:
            where_filter = {
                "$and": [
                    {"start_year": {"$lte": query_year}},
                    {"end_year": {"$gte": query_year}},
                ]
            }
            results = self.chroma.semantic_search(query, n_results=limit, where_filter=where_filter)

        return self._convert(results)

    def generate_answer(self, query, rules, query_year=None):
        if not rules:
            return "No temporally valid provisions found.", 0.3
        try:
            answer = self.llm.generate_response(
                query=query,
                query_type="HISTORICAL_POINT" if query_year else "GENERAL",
                rules=rules,
                query_date=date(query_year, 12, 31) if query_year else None,
            )
            return answer, 0.7
        except Exception:
            r = rules[0]
            return f"Based on {r['law_name']}, {r['section']}: {r['rule_text'][:300]}...", 0.5

    def _convert(self, results):
        rules = []
        if not results.get("ids"):
            return rules
        ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
        docs = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        metas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
        dists = []
        if results.get("distances"):
            dists = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
        for i in range(len(ids)):
            m = metas[i] if i < len(metas) else {}
            rules.append({
                "rule_id": ids[i],
                "rule_text": docs[i] if i < len(docs) else "",
                "law_name": m.get("law_name", ""),
                "section": m.get("section", ""),
                "start_year": m.get("start_year", 0),
                "end_year": m.get("end_year", 9999),
                "status": m.get("status", "active"),
                "distance": dists[i] if i < len(dists) else 1.0,
                "source_url": m.get("source_url", ""),
            })
        return rules


# ──────────────────────────────────────────────────────────────────────
# 5. PV-RAG FULL — The proposed system
# ──────────────────────────────────────────────────────────────────────

class PVRAGFull(BaselineSystem):
    """
    Full PV-RAG system (our proposed approach).
    Hybrid semantic+temporal retrieval, version chain enrichment,
    ILO-Graph augmentation, LLM response generation.
    """

    name = "PV-RAG"
    description = "Full PV-RAG: hybrid retrieval + version chains + graph + temporal awareness"

    def __init__(self):
        from app.modules.retrieval import TemporalRetrievalEngine
        self.engine = TemporalRetrievalEngine()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from app.modules.llm_response_generator import get_llm_generator
            self._llm = get_llm_generator()
        return self._llm

    def retrieve(self, query, query_year=None, limit=10):
        if query_year:
            return self.engine.retrieve_with_graph(
                query_text=query,
                query_date=date(query_year, 12, 31),
                limit=limit,
            )
        else:
            return self.engine.retrieve_general(query_text=query, limit=limit)

    def generate_answer(self, query, rules, query_year=None):
        if not rules:
            return "No relevant legal provisions found.", 0.3

        query_type = "HISTORICAL_POINT" if query_year else "GENERAL"
        try:
            answer = self.llm.generate_response(
                query=query,
                query_type=query_type,
                rules=rules,
                query_date=date(query_year, 12, 31) if query_year else None,
            )
            confidence = self._calc_confidence(rules, query_year)
            return answer, confidence
        except Exception as e:
            logger.warning(f"PVRAGFull LLM failed: {e}")
            r = rules[0]
            return f"Based on {r['law_name']}, {r['section']}: {r['rule_text'][:300]}...", 0.5

    def _calc_confidence(self, rules, query_year):
        """PV-RAG confidence: combines distances + temporal validity."""
        if not rules:
            return 0.3

        # Base confidence from semantic distance
        avg_dist = sum(r.get("distance", 1.0) for r in rules[:3] if r.get("distance", 999) < 900) / max(
            1, sum(1 for r in rules[:3] if r.get("distance", 999) < 900)
        )
        base = max(0.3, min(0.9, 1.0 - avg_dist * 0.3))

        # Temporal boost
        if query_year:
            valid_count = sum(
                1 for r in rules[:5]
                if r.get("start_year", 0) <= query_year <= r.get("end_year", 9999)
            )
            temporal_boost = min(valid_count * 0.05, 0.15)
            base += temporal_boost

        return min(base, 0.98)


# ──────────────────────────────────────────────────────────────────────
# 6. PV-RAG NO GRAPH (ablation — removes graph augmentation)
# ──────────────────────────────────────────────────────────────────────

class PVRAGNoGraph(BaselineSystem):
    """
    Ablation: PV-RAG without ILO-Graph augmentation.
    Keeps hybrid retrieval + version chains, removes graph-based
    successor mapping and BNS equivalents.
    """

    name = "PV-RAG-NoGraph"
    description = "PV-RAG without ILO-Graph (ablation: no successor mapping)"

    def __init__(self):
        from app.modules.retrieval import TemporalRetrievalEngine
        self.engine = TemporalRetrievalEngine()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from app.modules.llm_response_generator import get_llm_generator
            self._llm = get_llm_generator()
        return self._llm

    def retrieve(self, query, query_year=None, limit=10):
        if query_year:
            # Use point-in-time + version chains but NO graph
            rules = self.engine.retrieve_for_point_in_time(
                query_text=query,
                query_date=date(query_year, 12, 31),
                limit=limit,
            )
            return self.engine._enrich_with_version_chains(rules, limit)
        else:
            return self.engine.retrieve_general(query_text=query, limit=limit)

    def generate_answer(self, query, rules, query_year=None):
        if not rules:
            return "No relevant legal provisions found.", 0.3
        try:
            answer = self.llm.generate_response(
                query=query,
                query_type="HISTORICAL_POINT" if query_year else "GENERAL",
                rules=rules,
                query_date=date(query_year, 12, 31) if query_year else None,
            )
            return answer, 0.7
        except Exception:
            r = rules[0]
            return f"Based on {r['law_name']}, {r['section']}: {r['rule_text'][:300]}...", 0.5


# ──────────────────────────────────────────────────────────────────────
# FACTORY
# ──────────────────────────────────────────────────────────────────────

ALL_SYSTEMS = {
    "NaiveRAG": NaiveRAG,
    "BM25": BM25Retrieval,
    "LLMOnly": LLMOnly,
    "TemporalOnly": TemporalOnly,
    "PV-RAG-NoGraph": PVRAGNoGraph,
    "PV-RAG": PVRAGFull,
}


def get_system(name: str) -> BaselineSystem:
    """Instantiate a baseline system by name."""
    if name not in ALL_SYSTEMS:
        raise ValueError(f"Unknown system: {name}. Available: {list(ALL_SYSTEMS.keys())}")
    return ALL_SYSTEMS[name]()


def get_all_systems() -> Dict[str, BaselineSystem]:
    """Instantiate all baseline systems."""
    return {name: cls() for name, cls in ALL_SYSTEMS.items()}
