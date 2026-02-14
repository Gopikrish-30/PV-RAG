"""
Temporal Retrieval Engine (ChromaDB version)
Handles vector database queries with temporal filtering.

ChromaDB metadata fields (set by load_data_chromadb.py):
  rule_id, law_name, section, rule_topic, start_year, end_year (9999=active),
  status, source, source_url
"""
from typing import List, Dict, Optional
from datetime import date, datetime
from loguru import logger

from app.db.chromadb_manager import get_chroma_manager


class TemporalRetrievalEngine:
    """Retrieve legal rules with temporal awareness from ChromaDB"""

    def __init__(self):
        self.chroma = get_chroma_manager()

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    def retrieve_for_point_in_time(
        self,
        query_text: str,
        query_date: date,
        topics: List[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Retrieve rules relevant to a specific point in time (hybrid)."""
        logger.info(f"Retrieving rules for date: {query_date}, query: '{query_text[:80]}'")

        results = self.chroma.query_by_point_in_time(
            query_text=query_text,
            query_date=query_date,
            n_results=limit,
        )

        rules = self._convert_results(results)

        # Annotate each rule with temporal applicability
        qy = query_date.year
        for r in rules:
            sy = r.get("start_year", 0)
            ey = r.get("end_year", 9999)
            r["temporally_valid"] = (sy <= qy <= ey)

        logger.info(f"Found {len(rules)} rules ({sum(1 for r in rules if r.get('temporally_valid'))} temporally valid at {query_date})")
        return rules

    def retrieve_latest(
        self,
        query_text: str,
        topics: List[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Retrieve currently active rules (hybrid – active boosted)."""
        logger.info(f"Retrieving latest rules for: '{query_text[:80]}'")

        results = self.chroma.query_latest(
            query_text=query_text,
            n_results=limit,
        )

        rules = self._convert_results(results)

        # Annotate active status
        for r in rules:
            r["temporally_valid"] = (r.get("end_year", 9999) >= 9999)

        logger.info(f"Found {len(rules)} rules ({sum(1 for r in rules if r.get('temporally_valid'))} active)")
        return rules

    def retrieve_by_date_range(
        self,
        query_text: str,
        start_year: int,
        end_year: int,
        limit: int = 20,
    ) -> List[Dict]:
        """Retrieve rules overlapping with a year range."""
        results = self.chroma.query_by_date_range(
            query_text=query_text,
            start_year=start_year,
            end_year=end_year,
            n_results=limit,
        )
        return self._convert_results(results)

    def retrieve_general(
        self,
        query_text: str,
        limit: int = 10,
    ) -> List[Dict]:
        """Pure semantic search without temporal filters."""
        results = self.chroma.semantic_search(query_text, n_results=limit)
        return self._convert_results(results)

    # ------------------------------------------------------------------
    # Result conversion
    # ------------------------------------------------------------------

    def _convert_results(self, results: Dict) -> List[Dict]:
        """Convert raw ChromaDB results to a flat list of rule dicts."""
        rules: List[Dict] = []

        if not results.get("ids") or not results["ids"]:
            return rules

        # ChromaDB wraps in nested lists for query (not for get)
        ids = results["ids"][0] if results["ids"] and isinstance(results["ids"][0], list) else results["ids"]
        documents = results["documents"][0] if results["documents"] and isinstance(results["documents"][0], list) else results["documents"]
        metadatas = results["metadatas"][0] if results["metadatas"] and isinstance(results["metadatas"][0], list) else results["metadatas"]
        distances = []
        if results.get("distances"):
            distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            rule = {
                "rule_id": ids[i],
                "rule_text": documents[i] if i < len(documents) else "",
                # Normalise field names: always expose both naming conventions
                "law_name": meta.get("law_name", meta.get("act_title", "")),
                "act_title": meta.get("law_name", meta.get("act_title", "")),
                "section": meta.get("section", meta.get("section_id", "")),
                "section_id": meta.get("section", meta.get("section_id", "")),
                "rule_topic": meta.get("rule_topic", "general"),
                "start_year": meta.get("start_year", 0),
                "end_year": meta.get("end_year", 9999),
                "status": meta.get("status", "active"),
                "source": meta.get("source", ""),
                "source_url": meta.get("source_url", ""),
                "distance": distances[i] if i < len(distances) else 0.0,
            }
            rules.append(rule)

        return rules

    # ------------------------------------------------------------------
    # Timeline builder
    # ------------------------------------------------------------------

    def build_timeline_dict(self, rules: List[Dict]) -> List[Dict]:
        """Convert list of rules to timeline entries for the API response."""
        timeline = []

        for rule in sorted(rules, key=lambda r: r.get("start_year", 0)):
            start_year = rule.get("start_year", 0)
            end_year = rule.get("end_year", 9999)
            end_display = "Present" if end_year >= 9999 else str(end_year)

            text = rule.get("rule_text", "")
            preview = (text[:200] + "...") if len(text) > 200 else text

            entry = {
                "rule_id": rule.get("rule_id", ""),
                "period": f"{start_year}-{end_display}",
                "start_date": f"{start_year}-01-01",
                "end_date": f"{end_year}-12-31" if end_year < 9999 else None,
                "status": rule.get("status", "unknown"),
                "version": None,
                "amendment_ref": None,
                "rule_text_preview": preview,
            }
            timeline.append(entry)

        return timeline
