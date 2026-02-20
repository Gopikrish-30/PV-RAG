"""
Web Agent for Legal Information Retrieval & Verification

Two modes:
1. verify_and_compare() — traditional: compare dataset results with web results
2. web_only_answer()    — NEW: for LATEST/current queries, search the web
   with multi-query variations and generate an answer purely from web sources
   (no dataset mixing to avoid stale/misleading results)

Uses Tavily for intelligent web search and Groq LLM for answer synthesis.
"""
from typing import List, Dict, Optional
from datetime import datetime
import re
from loguru import logger

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily not available. Install: pip install tavily-python")

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from config.settings import settings


# =====================================================================
# Trusted Indian Legal Web Sources
# =====================================================================

# Government & official portals — highest trust
TRUSTED_GOV_DOMAINS = [
    "indiacode.nic.in",            # Official India Code — all Central Acts
    "legislative.gov.in",          # Legislative Department
    "egazette.gov.in",             # Official Gazette of India
    "egazette.nic.in",             # Official Gazette (NIC mirror)
    "rajyasabha.nic.in",           # Rajya Sabha — Bill texts
    "sansad.in",                   # Parliament of India
    "prsindia.org",                # PRS Legislative Research (authoritative)
    "main.sci.gov.in",             # Supreme Court of India
    "sci.gov.in",                  # Supreme Court (alternate)
    "doj.gov.in",                  # Department of Justice
    "legalaffairs.gov.in",         # Department of Legal Affairs
    "niti.gov.in",                 # NITI Aayog
]

# Ministry & department portals — domain-specific legal acts
TRUSTED_MINISTRY_DOMAINS = [
    "morth.nic.in",                # Ministry of Road Transport — Motor Vehicles Act
    "cbic.gov.in",                 # Central Board of Indirect Taxes — GST, Customs
    "incometaxindia.gov.in",       # Income Tax Department
    "incometax.gov.in",            # Income Tax (new portal)
    "labour.gov.in",               # Ministry of Labour — labour laws, wages
    "nrega.nic.in",                # MGNREGA official portal
    "rural.nic.in",                # Ministry of Rural Development
    "meity.gov.in",                # MeitY — IT Act, Data Protection
    "mca.gov.in",                  # Ministry of Corporate Affairs — Companies Act
    "consumeraffairs.nic.in",      # Consumer Affairs
    "wcd.nic.in",                  # Women & Child Development
    "moef.gov.in",                 # Ministry of Environment — environmental laws
    "mha.gov.in",                  # Ministry of Home Affairs — IPC/BNS
]

# Trusted legal news & analysis portals
TRUSTED_LEGAL_MEDIA = [
    "livelaw.in",                  # Live Law — legal news, judgments
    "barandbench.com",             # Bar and Bench — legal journalism
    "scconline.com",               # SCC Online — case law database
    "manupatra.com",               # Manupatra — legal database
    "indiankanoon.org",            # Indian Kanoon — free case law search
    "latestlaws.com",              # Latest Laws
    "legalbites.in",               # Legal Bites
]

# Combined: all trusted domains
ALL_TRUSTED_DOMAINS = TRUSTED_GOV_DOMAINS + TRUSTED_MINISTRY_DOMAINS + TRUSTED_LEGAL_MEDIA

# Domains to always exclude
EXCLUDED_DOMAINS = [
    "wikipedia.org",
    "quora.com",
    "reddit.com",
    "answers.yahoo.com",
    "brainly.in",
]


class WebVerificationAgent:
    """
    Agent that searches the web for legal information.
    Supports two modes:
      - verify_and_compare(): compare dataset + web (for historical queries with web check)
      - web_only_answer():    web-only pipeline for LATEST/current queries
    """

    def __init__(self):
        self.client = None
        self.llm = None

        # Tavily client
        if TAVILY_AVAILABLE and settings.tavily_api_key:
            try:
                self.client = TavilyClient(api_key=settings.tavily_api_key)
                logger.info("Tavily Web Agent initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily: {e}")

        # Groq LLM for synthesis
        if GROQ_AVAILABLE and settings.groq_api_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=settings.groq_api_key,
                    model_name=settings.groq_model,
                    temperature=0.1,
                    max_tokens=800,
                )
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # =================================================================
    # NEW: Web-Only Answer Pipeline (for LATEST / current queries)
    # =================================================================

    def web_only_answer(
        self,
        query: str,
        multi_queries: List[str],
    ) -> Dict:
        """
        Search the web using multiple query variations and generate
        an answer PURELY from web results. No dataset mixing.

        Args:
            query: original user query
            multi_queries: list of LLM-rewritten query variations

        Returns:
            {
                "answered": bool,
                "answer": str,
                "sources": List[str],
                "confidence": float,
                "web_snippets": str,
            }
        """
        if not self.enabled:
            logger.warning("Web agent not available for web-only answer")
            return self._no_web_answer("Web search is not available.")

        try:
            # --- Step 1: Search with multiple query variations ---
            all_results = []
            seen_urls = set()

            for i, search_q in enumerate(multi_queries[:5]):
                try:
                    logger.info(f"Web search [{i+1}/{min(len(multi_queries), 5)}]: {search_q[:80]}")
                    results = self.client.search(
                        query=search_q,
                        search_depth="advanced",
                        max_results=5,
                        include_domains=ALL_TRUSTED_DOMAINS,
                        exclude_domains=EXCLUDED_DOMAINS,
                    )
                    for r in results.get("results", []):
                        url = r.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                except Exception as e:
                    logger.warning(f"Search variation {i+1} failed: {e}")
                    continue

            if not all_results:
                logger.warning("No web results found for any query variation")
                return self._no_web_answer(
                    "I couldn't find relevant current legal information on the web. "
                    "Please try rephrasing your question with specific Act names or Sections."
                )

            # --- Step 2: Rank and deduplicate results ---
            ranked = self._rank_results(all_results)

            # Extract content
            sources: List[str] = []
            content_parts: List[str] = []
            for r in ranked[:10]:
                url = r.get("url", "")
                title = r.get("title", "")
                content = r.get("content", "")
                if url:
                    sources.append(url)
                domain = self._extract_domain(url)
                snippet = f"[{domain}] {title}\n{content[:500]}"
                content_parts.append(snippet)

            web_text = "\n\n---\n\n".join(content_parts)
            confidence = self._calculate_web_confidence(ranked, sources)

            logger.info(f"Web-only: {len(ranked)} unique results, confidence={confidence:.2f}")

            # --- Step 3: LLM synthesis (web-only, no dataset) ---
            if self.llm:
                answer = self._synthesize_web_only_answer(query, web_text, sources)
                if answer:
                    return {
                        "answered": True,
                        "answer": answer,
                        "sources": sources[:8],
                        "confidence": confidence,
                        "web_snippets": web_text[:2000],
                    }

            # Fallback: return raw web text if LLM fails
            return {
                "answered": True,
                "answer": f"Based on web search results:\n\n{web_text[:1500]}",
                "sources": sources[:8],
                "confidence": max(confidence - 0.1, 0.3),
                "web_snippets": web_text[:2000],
            }

        except Exception as e:
            logger.error(f"Web-only answer failed: {e}")
            return self._no_web_answer(f"Web search encountered an error: {str(e)}")

    def _synthesize_web_only_answer(
        self, query: str, web_text: str, sources: List[str]
    ) -> Optional[str]:
        """Use LLM to produce a clean answer from web-only results."""
        year = datetime.now().year

        gov_sources = [s for s in sources if any(d in s for d in
                       ["gov.in", "nic.in", "prsindia.org", "sansad.in"])]
        legal_sources = [s for s in sources if any(d in s for d in
                         ["livelaw.in", "barandbench.com", "indiankanoon.org",
                          "scconline.com", "manupatra.com"])]

        source_note = ""
        if gov_sources:
            source_note += f"\nGovernment sources found: {len(gov_sources)}"
        if legal_sources:
            source_note += f"\nLegal media sources found: {len(legal_sources)}"

        prompt = f"""Answer the user's legal question using ONLY the web search results below.
This is a query about CURRENT / LATEST Indian law (as of {year}).

User Question: {query}

=== WEB SEARCH RESULTS ===
{web_text[:3000]}
{source_note}

Instructions:
1. Provide the CURRENT legal position based on the web results.
2. Cite the specific Act name, Section number, penalty amount, and effective date.
3. If a new Act has replaced an old one (e.g., BNS replaced IPC), mention both.
4. If the web results mention recent amendments, include that information.
5. If the web results are insufficient or contradictory, say so honestly.
6. Do NOT invent information not present in the web results.
7. Keep response under 300 words.
8. Structure: Brief answer → Legal reference → Key details → Source note.
"""

        try:
            messages = [
                ("system",
                 "You are an expert Indian legal information assistant. "
                 "Answer ONLY from the provided web search results. "
                 "Always cite specific Acts, Sections, and years. "
                 "Prefer government sources over media sources. "
                 "Be precise about penalty amounts, dates, and legal provisions."),
                ("human", prompt),
            ]
            resp = self.llm.invoke(messages)
            text = resp.content.strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if len(text) > 50:
                return text
        except Exception as e:
            logger.error(f"Web-only LLM synthesis failed: {e}")
        return None

    def _rank_results(self, results: List[Dict]) -> List[Dict]:
        """Rank web results: government > legal media > others."""
        def score_result(r):
            url = r.get("url", "")
            base = r.get("score", 0.5)
            if any(d in url for d in TRUSTED_GOV_DOMAINS):
                base += 0.40
            elif any(d in url for d in TRUSTED_MINISTRY_DOMAINS):
                base += 0.35
            elif any(d in url for d in TRUSTED_LEGAL_MEDIA):
                base += 0.15
            if not any(d in url for d in ALL_TRUSTED_DOMAINS):
                base -= 0.10
            return base
        return sorted(results, key=score_result, reverse=True)

    def _calculate_web_confidence(self, results: List[Dict], sources: List[str]) -> float:
        """Calculate confidence score for web-only results."""
        confidence = 0.4
        gov_count = sum(1 for s in sources if any(d in s for d in ["gov.in", "nic.in"]))
        confidence += min(gov_count * 0.10, 0.25)
        legal_count = sum(1 for s in sources if any(
            d in s for d in ["livelaw.in", "barandbench.com", "indiankanoon.org",
                             "scconline.com", "prsindia.org"]))
        confidence += min(legal_count * 0.05, 0.10)
        confidence += min(len(results) * 0.02, 0.10)
        if results:
            avg_score = sum(r.get("score", 0) for r in results) / len(results)
            confidence += avg_score * 0.10
        return min(confidence, 0.95)

    def _extract_domain(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return "web"

    def _no_web_answer(self, message: str = "") -> Dict:
        return {
            "answered": False,
            "answer": message or "Could not find current legal information on the web.",
            "sources": [],
            "confidence": 0.0,
            "web_snippets": "",
        }

    # =================================================================
    # EXISTING: Verify & Compare (dataset + web)
    # =================================================================

    def verify_and_compare(
        self,
        query: str,
        dataset_rules: List[Dict],
    ) -> Dict:
        """
        Search the web for the query and compare with dataset results.

        Returns:
            {
                "verified": bool,
                "web_answer": str,
                "combined_answer": str,
                "sources": List[str],
                "confidence": float,
            }
        """
        if not self.enabled:
            return self._no_result()

        try:
            primary = dataset_rules[0] if dataset_rules else {}
            search_q = self._build_search_query(query, primary)
            logger.info(f"Tavily search: {search_q}")

            results = self.client.search(
                query=search_q,
                search_depth="advanced",
                max_results=settings.max_search_results,
                include_domains=ALL_TRUSTED_DOMAINS,
                exclude_domains=EXCLUDED_DOMAINS,
            )

            if not results or not results.get("results"):
                logger.warning("No web results found")
                return self._no_result()

            sources: List[str] = []
            content_parts: List[str] = []
            for r in results.get("results", [])[:5]:
                url = r.get("url", "")
                title = r.get("title", "")
                content = r.get("content", "")
                if url:
                    sources.append(url)
                snippet = f"{title}: {content[:400]}" if title else content[:400]
                content_parts.append(snippet)

            web_text = "\n\n".join(content_parts)
            confidence = self._calculate_confidence(results, sources)

            combined = web_text
            if self.llm and dataset_rules:
                combined = self._llm_compare(query, dataset_rules, web_text)

            return {
                "verified": True,
                "web_answer": web_text[:1000],
                "combined_answer": combined,
                "sources": sources[:5],
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Web verification failed: {e}")
            return self._no_result()

    def verify_latest_information(
        self,
        query: str,
        act_title: Optional[str] = None,
        section_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Dict:
        """Legacy-compatible wrapper."""
        return self.verify_and_compare(query, [
            {"law_name": act_title or "", "section": section_id or "", "rule_topic": topic or ""}
        ])

    # -----------------------------------------------------------------
    # Decision logic
    # -----------------------------------------------------------------

    def judge_need_for_web_verification(
        self,
        query: str,
        query_type: str,
        retrieved_rules: List[Dict],
        query_date=None,
    ) -> bool:
        """
        Decide whether web verification is needed.
        NOTE: LATEST queries are now routed to web_only_answer() in main.py,
        so this is mainly for GENERAL / COMPARISON queries that might benefit
        from a web cross-check.
        """
        if not self.enabled:
            return False

        if query_type == "HISTORICAL_POINT":
            return False

        # LATEST is handled by web-only pipeline
        if query_type == "LATEST":
            return False

        if not retrieved_rules:
            return True

        current_year = datetime.now().year
        primary = retrieved_rules[0]
        end_year = primary.get("end_year", 9999)
        if isinstance(end_year, str):
            try:
                end_year = int(end_year.split("-")[0])
            except Exception:
                end_year = 9999
        if end_year < current_year - 2:
            return True

        latest_kw = ["current", "latest", "now", "today", "present", "recent"]
        if any(kw in query.lower() for kw in latest_kw):
            return True

        return False

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _build_search_query(self, query: str, primary: Dict) -> str:
        parts = [query]
        law = primary.get("law_name") or primary.get("act_title", "")
        if law:
            parts.append(f'"{law}"')
        section = primary.get("section") or primary.get("section_id", "")
        if section:
            parts.append(f"Section {section}")
        parts.append("latest India")
        parts.append(str(datetime.now().year))
        return " ".join(parts)[:500]

    def _llm_compare(self, query: str, rules: List[Dict], web_text: str) -> str:
        """Use the LLM to compare dataset answer vs web answer."""
        # Summarise dataset rules
        dataset_summary = ""
        for i, r in enumerate(rules[:3], 1):
            dataset_summary += (
                f"Rule {i}: {r.get('law_name', r.get('act_title', 'N/A'))} "
                f"Section {r.get('section', r.get('section_id', 'N/A'))} "
                f"({r.get('start_year', '?')}-{r.get('end_year', 'Present')}) "
                f"Status: {r.get('status', '?')}\n"
                f"Text: {r.get('rule_text', '')[:300]}\n\n"
            )

        prompt = f"""Compare the DATASET answer with the WEB answer for the user's question and provide the most accurate, up-to-date response.

User Question: {query}

=== DATASET (offline legal database) ===
{dataset_summary}

=== WEB (live search results) ===
{web_text[:1500]}

Instructions:
1. If both agree, confirm the answer with high confidence.
2. If they disagree, prefer the more recent / authoritative source.
3. Cite specific Acts, Sections, and years.
4. Keep response under 200 words.
"""
        try:
            messages = [
                ("system", "You are an Indian legal information assistant. Synthesise dataset and web results accurately."),
                ("human", prompt),
            ]
            resp = self.llm.invoke(messages)
            text = resp.content.strip()
            # Strip <think>...</think> reasoning tags (qwen3 thinking model)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as e:
            logger.error(f"LLM compare failed: {e}")
            return ""

    def _calculate_confidence(self, results: Dict, sources: List[str]) -> float:
        confidence = 0.5
        official = ["gov.in", "nic.in"]
        official_count = sum(1 for s in sources if any(d in s for d in official))
        confidence += official_count * 0.12
        n = len(results.get("results", []))
        confidence += min(n * 0.04, 0.16)
        avg_score = sum(r.get("score", 0) for r in results.get("results", [])) / max(n, 1)
        confidence += avg_score * 0.1
        return min(confidence, 0.95)

    def _no_result(self) -> Dict:
        return {
            "verified": False,
            "web_answer": "",
            "combined_answer": "",
            "sources": [],
            "confidence": 0.0,
        }


# Singleton
_web_agent = None


def get_web_agent() -> WebVerificationAgent:
    global _web_agent
    if _web_agent is None:
        _web_agent = WebVerificationAgent()
    return _web_agent


def reset_web_agent():
    """Force re-creation of the web agent (picks up new API keys)."""
    global _web_agent
    _web_agent = None
    return get_web_agent()
