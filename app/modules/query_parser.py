"""
Query Understanding Module (LLM-powered + regex fallback)

Uses Groq LLM to understand user queries intelligently:
- Detects query type (historical, latest, general, comparison, range)
- Extracts temporal entities (years, dates)
- Extracts legal topics & rewrites query for better vector retrieval
- Falls back to regex when LLM is unavailable
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import re
import json
from loguru import logger

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from config.settings import settings


class QueryParser:
    """Parse and understand legal queries using LLM + regex fallback"""

    def __init__(self):
        """Initialize with optional LLM"""
        self.llm = None
        if GROQ_AVAILABLE and settings.groq_api_key:
            try:
                self.llm = ChatGroq(
                    groq_api_key=settings.groq_api_key,
                    model_name=settings.groq_model,
                    temperature=0.0,
                    max_tokens=400,
                )
                logger.info("QueryParser: LLM-powered understanding enabled")
            except Exception as e:
                logger.warning(f"QueryParser: LLM init failed ({e}), using regex")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def parse(self, query: str) -> Dict:
        """
        Parse a user query and return structured understanding.

        Returns dict with keys:
            original_query, query_type, temporal_entities,
            legal_topics, search_query (rewritten for vector DB),
            needs_web_verification
        """
        # Greeting / off-topic check FIRST (before wasting an LLM call)
        if self._is_greeting(query):
            logger.info(f"Detected greeting/off-topic: '{query}'")
            return {
                "original_query": query,
                "query_type": "GREETING",
                "temporal_entities": {},
                "legal_topics": [],
                "search_query": query,
                "needs_web_verification": False,
            }

        if self.llm:
            try:
                return self._parse_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM parse failed ({e}), falling back to regex")

        return self._parse_with_regex(query)

    # ------------------------------------------------------------------
    # LLM-powered parsing
    # ------------------------------------------------------------------

    def _parse_with_llm(self, query: str) -> Dict:
        """Use Groq LLM to understand the query."""

        system_prompt = """You are a legal query understanding system for Indian law.
Given a user query, output ONLY valid JSON (no markdown, no extra text) with these exact keys:

{
  "query_type": "HISTORICAL_POINT" | "LATEST" | "DATE_RANGE" | "COMPARISON" | "GENERAL",
  "year": null or integer (for HISTORICAL_POINT),
  "start_year": null or integer (for DATE_RANGE),
  "end_year": null or integer (for DATE_RANGE),
  "legal_topics": ["topic1", "topic2"],
  "search_query": "rewritten query optimised for semantic search against Indian legal acts and sections"
}

Rules:
- If the user mentions a specific year or "in YYYY", set query_type to HISTORICAL_POINT and year to that integer.
- If the user says "current", "now", "today", "latest", "present" or asks without specifying a date, set query_type to LATEST.
- If the user mentions two years as a range, use DATE_RANGE.
- If the user asks to compare two time periods, use COMPARISON.
- Otherwise use GENERAL.
- legal_topics: extract key legal concepts (e.g. "motor vehicles", "income tax", "penalty", "copyright").
- search_query: rewrite the question into keywords and phrases that would match Indian legal act text in a vector database. Include act names if obvious (e.g. "Motor Vehicles Act" for helmet questions). DO NOT include years in search_query.
- If the query is a casual greeting or off-topic, still set query_type to GENERAL and search_query to the original text."""

        messages = [
            ("system", system_prompt),
            ("human", query),
        ]

        response = self.llm.invoke(messages)
        raw = response.content.strip()

        # Strip <think>...</think> reasoning tags (qwen3 thinking model)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed_json = json.loads(raw)

        # Build the result dict
        query_type = parsed_json.get("query_type", "GENERAL")
        temporal_entities: Dict = {}

        if query_type == "HISTORICAL_POINT" and parsed_json.get("year"):
            yr = int(parsed_json["year"])
            temporal_entities = {"year": yr, "date": date(yr, 12, 31)}

        elif query_type == "DATE_RANGE":
            sy = parsed_json.get("start_year")
            ey = parsed_json.get("end_year")
            if sy and ey:
                temporal_entities = {"start_year": int(sy), "end_year": int(ey)}

        elif query_type == "LATEST":
            temporal_entities = {"date": date.today()}

        elif query_type == "COMPARISON":
            y1 = parsed_json.get("start_year") or parsed_json.get("year")
            y2 = parsed_json.get("end_year")
            if y1 and y2:
                temporal_entities = {"year1": int(y1), "year2": int(y2)}

        search_query = parsed_json.get("search_query", query)
        topics = parsed_json.get("legal_topics", ["general"])
        if not topics:
            topics = ["general"]

        result = {
            "original_query": query,
            "query_type": query_type,
            "temporal_entities": temporal_entities,
            "legal_topics": topics,
            "search_query": search_query,
            "needs_web_verification": query_type == "LATEST",
        }
        logger.info(f"LLM parsed → type={query_type}, topics={topics}, temporal={temporal_entities}, search='{search_query[:60]}'")
        return result

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    HISTORICAL_PATTERNS = [
        r'(?:what was|how was|tell me about).*?(?:in\s+)?(\d{4})',
        r'(\d{4}).*?(?:what|how|tell)',
        r'back in (\d{4})',
        r'during (\d{4})',
        r'as of (\d{4})',
        r'year (\d{4})',
    ]

    LATEST_PATTERNS = [
        r'\b(?:current|latest|now|today|present|existing|prevailing)\b',
        r'\bwhat is\b',
        r'\bwhat are\b',
    ]

    RANGE_PATTERNS = [
        r'from (\d{4}) to (\d{4})',
        r'between (\d{4}) and (\d{4})',
    ]

    COMPARISON_PATTERNS = [
        r'compare.*?(\d{4}).*?(\d{4})',
        r'(\d{4})\s*(?:vs|versus|compared to|and)\s*(\d{4})',
        r'difference.*?(\d{4}).*?(\d{4})',
    ]

    TOPIC_KEYWORDS = {
        'helmet': 'motor vehicles',
        'traffic': 'motor vehicles',
        'motor': 'motor vehicles',
        'vehicle': 'motor vehicles',
        'driving': 'motor vehicles',
        'drive': 'motor vehicles',
        'bike': 'motor vehicles',
        'ridebike': 'motor vehicles',
        'motorcycle': 'motor vehicles',
        'scooter': 'motor vehicles',
        'overspeeding': 'motor vehicles',
        'overspeed': 'motor vehicles',
        'speed': 'motor vehicles',
        'drunk': 'motor vehicles',
        'drunken': 'motor vehicles',
        'seatbelt': 'motor vehicles',
        'fine': 'penalty',
        'penalty': 'penalty',
        'punishment': 'criminal law',
        'tax': 'taxation',
        'income tax': 'income tax',
        'gst': 'goods and services tax',
        'customs': 'customs',
        'copyright': 'copyright',
        'patent': 'patent',
        'trademark': 'trademark',
        'contract': 'contracts',
        'employment': 'labor',
        'labour': 'labor',
        'wage': 'wages',
        'salary': 'wages',
        'data protection': 'data protection',
        'privacy': 'privacy',
        'consumer': 'consumer protection',
        'marriage': 'marriage',
        'divorce': 'marriage',
        'company': 'companies',
        'corporate': 'companies',
        'ipc': 'indian penal code',
        'crpc': 'criminal procedure',
        'cpc': 'civil procedure',
        'environment': 'environment',
        'railway': 'railways',
        'army': 'armed forces',
    }

    GREETING_PATTERNS = [
        r'^\s*(hi|hello|hey|howdy|hola|greetings|good\s*(morning|evening|afternoon|day)|sup|yo|namaste)\s*[!.?]*\s*$',
        r'^\s*(thanks|thank you|thx|ok|okay|bye|goodbye|see ya)\s*[!.?]*\s*$',
    ]

    def _is_greeting(self, query: str) -> bool:
        """Check if query is a casual greeting or off-topic short message."""
        for pattern in self.GREETING_PATTERNS:
            if re.match(pattern, query.strip(), re.I):
                return True
        # Very short non-legal messages (1-2 words, no legal keywords)
        words = query.strip().split()
        if len(words) <= 2:
            topics = self._extract_topics(query.lower())
            # _extract_topics returns ["general"] when nothing matches
            if topics == ["general"]:
                return True
        return False

    def _build_search_query(self, query: str, topics: List[str]) -> str:
        """Rewrite query for better vector DB matching (regex fallback)."""
        # Strip temporal words and filler to get the core legal question
        cleaned = re.sub(r'\b(what|was|is|are|the|in|of|for|a|an|how|tell|me|about|please|can|you|current|latest|now|today|present)\b', ' ', query, flags=re.I)
        cleaned = re.sub(r'\b\d{4}\b', ' ', cleaned)  # remove years
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # If cleaning ate everything, fall back to original
        if len(cleaned) < 3:
            cleaned = query
        # Append act names from topic detection
        topic_to_act = {
            'motor vehicles': 'Motor Vehicles Act penalty',
            'income tax': 'Income Tax Act',
            'taxation': 'Tax Act',
            'criminal law': 'Indian Penal Code punishment',
            'companies': 'Companies Act',
            'consumer protection': 'Consumer Protection Act',
            'criminal procedure': 'Code of Criminal Procedure',
            'civil procedure': 'Code of Civil Procedure',
            'indian penal code': 'Indian Penal Code',
        }
        extras = []
        for t in topics:
            if t in topic_to_act:
                extras.append(topic_to_act[t])
        if extras:
            cleaned = cleaned + ' ' + ' '.join(extras)
        return cleaned.strip()

    def _parse_with_regex(self, query: str) -> Dict:
        """Regex-based fallback parser."""
        query_lower = query.lower().strip()

        # Greeting / off-topic detection
        if self._is_greeting(query_lower):
            logger.info(f"Detected greeting/off-topic: '{query}'")
            return {
                "original_query": query,
                "query_type": "GREETING",
                "temporal_entities": {},
                "legal_topics": [],
                "search_query": query,
                "needs_web_verification": False,
            }

        query_type, temporal = self._detect_query_type(query_lower)
        topics = self._extract_topics(query_lower)
        search_query = self._build_search_query(query, topics)

        result = {
            "original_query": query,
            "query_type": query_type,
            "temporal_entities": temporal,
            "legal_topics": topics,
            "search_query": search_query,
            "needs_web_verification": query_type == "LATEST",
        }

        logger.info(f"Regex parsed → type={query_type}, topics={topics}, temporal={temporal}, search='{search_query[:60]}'")
        return result

    def _detect_query_type(self, query: str) -> Tuple[str, Dict]:
        for pattern in self.COMPARISON_PATTERNS:
            m = re.search(pattern, query, re.I)
            if m:
                return "COMPARISON", {"year1": int(m.group(1)), "year2": int(m.group(2))}

        for pattern in self.RANGE_PATTERNS:
            m = re.search(pattern, query, re.I)
            if m:
                return "DATE_RANGE", {"start_year": int(m.group(1)), "end_year": int(m.group(2))}

        for pattern in self.HISTORICAL_PATTERNS:
            m = re.search(pattern, query, re.I)
            if m:
                yr = int(m.group(1))
                return "HISTORICAL_POINT", {"year": yr, "date": date(yr, 12, 31)}

        for pattern in self.LATEST_PATTERNS:
            if re.search(pattern, query, re.I):
                return "LATEST", {"date": date.today()}

        # Any standalone year
        m = re.search(r'\b(1[89]\d{2}|20\d{2})\b', query)
        if m:
            yr = int(m.group(1))
            return "HISTORICAL_POINT", {"year": yr, "date": date(yr, 12, 31)}

        return "GENERAL", {}

    def _extract_topics(self, query: str) -> List[str]:
        topics = []
        for keyword, topic in self.TOPIC_KEYWORDS.items():
            if keyword in query and topic not in topics:
                topics.append(topic)
        return topics if topics else ["general"]
