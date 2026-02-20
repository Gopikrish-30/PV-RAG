"""
LLM Response Generator using Groq
Generates natural language responses from retrieved legal rules.
Uses field names: law_name, section, start_year, end_year, status, rule_text
"""
from typing import List, Dict, Optional
from datetime import date
import re
from loguru import logger

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq/LangChain not available. Install: pip install langchain-groq")

from config.settings import settings


class LLMResponseGenerator:
    """Generate natural language responses using Groq LLM"""

    def __init__(self):
        self.llm = None
        self.enabled = settings.use_llm_for_response and GROQ_AVAILABLE
        self._last_source = "template"  # tracks which path generated the last answer

        if self.enabled:
            if not settings.groq_api_key:
                logger.warning("GROQ_API_KEY not set. Using template-based responses.")
                self.enabled = False
            else:
                try:
                    self.llm = ChatGroq(
                        groq_api_key=settings.groq_api_key,
                        model_name=settings.groq_model,
                        temperature=0.1,
                        max_tokens=600,
                    )
                    logger.info(f"Groq LLM initialized: {settings.groq_model}")
                except Exception as e:
                    logger.error(f"Failed to initialize Groq: {e}")
                    self.enabled = False

    def generate_response(
        self,
        query: str,
        query_type: str,
        rules: List[Dict],
        query_date: Optional[date] = None,
        web_result: Optional[Dict] = None,
    ) -> str:
        """
        Generate a natural language answer from retrieved rules
        (optionally enriched with web verification).
        """
        # If web agent produced a combined answer, use that
        if web_result and web_result.get("combined_answer"):
            self._last_source = "llm_web_merged"  # Path A
            return web_result["combined_answer"]

        if not self.enabled or not self.llm:
            self._last_source = "template"  # Path C
            return self._template_response(query, query_type, rules, query_date)

        try:
            context = self._build_context(rules, limit=5)
            prompt = self._build_prompt(query, query_type, context, query_date, web_result)

            messages = [
                (
                    "system",
                    "You are an expert Indian legal information assistant. "
                    "Provide accurate, concise answers based ONLY on the provided legal rules. "
                    "Always cite the specific Act name, Section number, and effective period. "
                    "If information is insufficient, say so honestly.",
                ),
                ("human", prompt),
            ]

            response = self.llm.invoke(messages)
            answer = response.content.strip()
            # Strip <think>...</think> reasoning tags (qwen3 thinking model)
            # Also handle unclosed <think> tags (model may not close them)
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL).strip()
            logger.info(f"LLM response generated ({len(answer)} chars)")
            # Path B: LLM with dataset rules (+ web context if available)
            self._last_source = "llm_dataset+web" if (web_result and web_result.get("web_answer")) else "llm_dataset"
            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            self._last_source = "template"  # Path C fallback
            return self._template_response(query, query_type, rules, query_date)

    # -----------------------------------------------------------------

    def _build_context(self, rules: List[Dict], limit: int = 5) -> str:
        parts = []
        for i, rule in enumerate(rules[:limit], 1):
            law = rule.get("law_name", rule.get("act_title", "N/A"))
            sec = rule.get("section", rule.get("section_id", "N/A"))
            sy = rule.get("start_year", "?")
            ey = rule.get("end_year", 9999)
            ey_display = "Present (active)" if ey >= 9999 else str(ey)
            status = rule.get("status", "unknown")
            text = rule.get("rule_text", "")[:500]

            parts.append(
                f"Rule {i}:\n"
                f"  Act: {law}\n"
                f"  Section: {sec}\n"
                f"  Valid: {sy} to {ey_display}\n"
                f"  Status: {status}\n"
                f"  Content: {text}\n"
            )
        return "\n".join(parts)

    def _build_prompt(
        self,
        query: str,
        query_type: str,
        context: str,
        query_date: Optional[date],
        web_result: Optional[Dict] = None,
    ) -> str:
        prompt = f"User Question: {query}\n\nRetrieved Legal Rules:\n{context}\n\n"

        if web_result and web_result.get("web_answer"):
            prompt += f"Web Search Results:\n{web_result['web_answer'][:800]}\n\n"

        if query_type == "HISTORICAL_POINT":
            year = query_date.year if query_date else "the specified time"
            prompt += (
                f"This is a HISTORICAL query about rules valid in {year}.\n"
                "1. State what was applicable in that year.\n"
                "2. Cite Act, Section, and effective period.\n"
                "3. Mention if the rule has since changed.\n"
                "4. Keep response under 200 words.\n"
            )
        elif query_type == "LATEST":
            prompt += (
                "This query asks for the CURRENT legal position.\n"
                "1. State the currently valid provision.\n"
                "2. Cite Act, Section, and effective date.\n"
                "3. If web results are available, incorporate them.\n"
                "4. Keep response under 200 words.\n"
            )
        elif query_type == "DATE_RANGE":
            prompt += (
                "This query asks about rules across a DATE RANGE.\n"
                "1. Summarise what changed over the period.\n"
                "2. Cite Acts and Sections.\n"
                "3. Keep response under 250 words.\n"
            )
        else:
            prompt += (
                "Provide relevant legal information from the retrieved rules.\n"
                "Cite specific Acts and Sections. Keep response under 200 words.\n"
            )

        return prompt

    # -----------------------------------------------------------------

    def _template_response(
        self,
        query: str,
        query_type: str,
        rules: List[Dict],
        query_date: Optional[date],
    ) -> str:
        """Rich fallback when LLM is unavailable – shows top 3 results."""
        if not rules:
            return "No relevant legal rules found for your query."

        parts: list = []

        # Primary result – detailed
        r = rules[0]
        law = r.get("law_name", r.get("act_title", "Unknown Act"))
        sec = r.get("section", r.get("section_id", "N/A"))
        sy = r.get("start_year", "Unknown")
        ey = r.get("end_year", 9999)
        ey_display = "Present" if ey >= 9999 else str(ey)
        status = r.get("status", "active")
        text_preview = r.get("rule_text", "")[:400]
        temporally_valid = r.get("temporally_valid")

        if query_type == "HISTORICAL_POINT":
            year = query_date.year if query_date else "the specified time"
            validity_note = ""
            if temporally_valid is False:
                validity_note = (
                    f"\n\n⚠ Note: This provision's recorded period is {sy}–{ey_display}, "
                    f"which may not exactly cover {year}. The rule text is the closest "
                    f"semantic match in the database."
                )
            parts.append(
                f"**As of {year}**, the most relevant provision is under **{law}**, "
                f"Section {sec} (valid {sy}–{ey_display}, status: {status}).\n\n"
                f"> {text_preview}"
                f"{validity_note}"
            )
        elif query_type == "LATEST":
            parts.append(
                f"**Current provision**: **{law}**, Section {sec} "
                f"(effective since {sy}, status: {status}).\n\n"
                f"> {text_preview}"
            )
        else:
            parts.append(
                f"**{law}**, Section {sec} ({sy}–{ey_display}).\n\n"
                f"> {text_preview}"
            )

        # Additional results (2nd and 3rd) – brief summaries
        if len(rules) > 1:
            parts.append("\n\n---\n**Other related provisions:**")
            for i, r2 in enumerate(rules[1:3], 2):
                law2 = r2.get("law_name", r2.get("act_title", "Unknown"))
                sec2 = r2.get("section", r2.get("section_id", ""))
                sy2 = r2.get("start_year", "?")
                ey2 = r2.get("end_year", 9999)
                ey2d = "Present" if ey2 >= 9999 else str(ey2)
                short_text = r2.get("rule_text", "")[:150]
                parts.append(
                    f"\n{i}. **{law2}** – Section {sec2} ({sy2}–{ey2d}): {short_text}..."
                )

        parts.append(f"\n\n*Found {len(rules)} total provision(s) in the database.*")
        return "\n".join(parts)


# Singleton
_generator = None


def get_llm_generator() -> LLMResponseGenerator:
    global _generator
    if _generator is None:
        _generator = LLMResponseGenerator()
    return _generator


def reset_llm_generator():
    """Force re-creation of the LLM generator (picks up new API keys)."""
    global _generator
    _generator = None
    return get_llm_generator()
