"""
Web Agent for Latest Legal Information Verification
Uses Tavily for intelligent web search and content extraction.
When enabled, fetches current info from the web and lets the LLM
compare it with the dataset results to produce the most accurate answer.
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


class WebVerificationAgent:
    """
    Agent that verifies latest legal information using web search.
    Uses Tavily for search + Groq LLM to synthesise dataset vs web.
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

        # Groq LLM for comparison / synthesis
        if GROQ_AVAILABLE and settings.groq_api_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=settings.groq_api_key,
                    model_name=settings.groq_model,
                    temperature=0.1,
                    max_tokens=500,
                )
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        return self.client is not None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

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
                "combined_answer": str,   # LLM synthesised
                "sources": List[str],
                "confidence": float,
            }
        """
        if not self.enabled:
            return self._no_result()

        try:
            # Determine search query from dataset context
            primary = dataset_rules[0] if dataset_rules else {}
            search_q = self._build_search_query(query, primary)
            logger.info(f"Tavily search: {search_q}")

            results = self.client.search(
                query=search_q,
                search_depth="advanced",
                max_results=settings.max_search_results,
                include_domains=[
                    "gov.in", "indiacode.nic.in", "egazette.nic.in",
                    "cbic.gov.in", "incometax.gov.in",
                    "legislative.gov.in", "livelaw.in", "barandbench.com",
                ],
                exclude_domains=["wikipedia.org", "quora.com"],
            )

            if not results or not results.get("results"):
                logger.warning("No web results found")
                return self._no_result()

            # Extract data
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

            # LLM comparison
            combined = web_text  # fallback
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
        Decide whether web verification is needed:
        - Historical queries: NO (dataset is authoritative)
        - Latest / General queries when the user didn't specify a year: YES
        - Stale data (rule ended long ago): YES
        """
        if not self.enabled:
            return False

        if query_type == "HISTORICAL_POINT":
            return False

        # If no rules were retrieved, try web
        if not retrieved_rules:
            return True

        # "Latest" queries always verify
        if query_type == "LATEST":
            return True

        # Check staleness
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

        # Keywords
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
