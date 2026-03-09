"""
Temporal Retrieval Engine (ChromaDB version)
Handles vector database queries with temporal filtering.

ChromaDB metadata fields (set by load_data_chromadb.py):
  rule_id, law_name, section, rule_topic, start_year, end_year (9999=active),
  status, source, source_url
"""
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime
from loguru import logger

from app.db.chromadb_manager import get_chroma_manager
from app.db.graph_manager import GraphManager


class TemporalRetrievalEngine:
    """Retrieve legal rules with temporal awareness from ChromaDB + ILO-Graph"""

    def __init__(self):
        self.chroma = get_chroma_manager()
        self.graph = GraphManager()

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    def retrieve_with_graph(
        self,
        query_text: str,
        query_date: date,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Hybrid retrieval: semantic search + graph-based temporal extension.

        1. Base semantic+temporal retrieval from ChromaDB
        2. For the top result's law+section, fetch all version chains from ChromaDB
        3. Graph augmentation: IPC->BNS successor mapping for post-2024 queries
        4. Annotate each rule with temporal validity
        """
        logger.info(f"Graph-Augmented retrieval for date: {query_date}")

        # 1. Base retrieval
        rules = self.retrieve_for_point_in_time(query_text, query_date, limit=limit)

        if not rules:
            return rules

        # 2. Version-chain enrichment: fetch sister versions from ChromaDB
        rules = self._enrich_with_version_chains(rules, limit)

        # 3. Graph augmentation: successor act mapping
        if self.graph.available:
            rules = self._augment_with_graph(rules, query_date)

        return rules

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
        self._annotate_temporal_validity(rules, query_date.year)

        logger.info(
            f"Found {len(rules)} rules "
            f"({sum(1 for r in rules if r.get('temporally_valid'))} temporally valid at {query_date})"
        )
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
        for r in rules:
            r["temporally_valid"] = r.get("end_year", 9999) >= 9999

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
        rules = self._convert_results(results)
        # Enrich with version chains so we get multi-version timelines
        rules = self._enrich_with_version_chains(rules, limit)
        return rules

    def retrieve_for_comparison(
        self,
        query_text: str,
        year1: int,
        year2: int,
        limit: int = 10,
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve rules for two specific years for comparison.
        Returns {"year1_rules": [...], "year2_rules": [...], "all_versions": [...]}.
        """
        logger.info(f"Comparison retrieval: {year1} vs {year2}")

        date1 = date(year1, 12, 31)
        date2 = date(year2, 12, 31)

        rules_y1 = self.retrieve_for_point_in_time(query_text, date1, limit=limit)
        rules_y2 = self.retrieve_for_point_in_time(query_text, date2, limit=limit)

        # Collect all version chains for the primary matched law+section
        all_rules = rules_y1 + rules_y2
        all_versions = self._enrich_with_version_chains(all_rules, limit * 2)

        return {
            "year1": year1,
            "year2": year2,
            "year1_rules": rules_y1,
            "year2_rules": rules_y2,
            "all_versions": all_versions,
        }

    def retrieve_general(
        self,
        query_text: str,
        limit: int = 10,
    ) -> List[Dict]:
        """Pure semantic search without temporal filters."""
        results = self.chroma.semantic_search(query_text, n_results=limit)
        return self._convert_results(results)

    # ------------------------------------------------------------------
    # Version-chain enrichment
    # ------------------------------------------------------------------

    def _enrich_with_version_chains(
        self,
        rules: List[Dict],
        limit: int = 20,
    ) -> List[Dict]:
        """
        For the top result(s), fetch all versions of the same law+section
        from ChromaDB to build a complete version timeline.
        Only enriches the primary law to avoid pulling in irrelevant acts.
        """
        if not rules:
            return rules

        # Determine the primary law to focus enrichment
        primary_law = self.get_primary_law(rules)

        # Collect unique (law_name, section) pairs from top results
        seen_pairs: set = set()
        chain_rules: Dict[str, Dict] = {}  # rule_id -> rule dict

        # Keep all original rules
        for r in rules:
            chain_rules[r["rule_id"]] = r

        # Fetch version chains for top 3 unique law+section pairs
        # from the primary law only
        for r in rules:
            law = r.get("law_name", "")
            # Only enrich primary law sections
            if primary_law and law != primary_law:
                continue

            key = (law, r.get("section", ""))
            if key in seen_pairs or not key[0] or not key[1]:
                continue
            seen_pairs.add(key)

            if len(seen_pairs) > 3:
                break

            try:
                timeline_results = self.chroma.query_timeline(
                    law_name=key[0], section=key[1]
                )
                if timeline_results and timeline_results.get("ids"):
                    ids = timeline_results["ids"]
                    docs = timeline_results.get("documents", [])
                    metas = timeline_results.get("metadatas", [])

                    for i, rid in enumerate(ids):
                        if rid not in chain_rules:
                            meta = metas[i] if i < len(metas) else {}
                            chain_rules[rid] = {
                                "rule_id": rid,
                                "rule_text": docs[i] if i < len(docs) else "",
                                "law_name": meta.get("law_name", key[0]),
                                "act_title": meta.get("law_name", key[0]),
                                "section": meta.get("section", key[1]),
                                "section_id": meta.get("section", key[1]),
                                "rule_topic": meta.get("rule_topic", "general"),
                                "start_year": meta.get("start_year", 0),
                                "end_year": meta.get("end_year", 9999),
                                "status": meta.get("status", "active"),
                                "source": meta.get("source", ""),
                                "source_url": meta.get("source_url", ""),
                                "distance": 999.0,  # didn't come from semantic search
                                "from_version_chain": True,
                            }
            except Exception as e:
                logger.debug(f"Version chain fetch failed for {key}: {e}")

        enriched = list(chain_rules.values())
        logger.info(f"Version-chain enrichment: {len(rules)} → {len(enriched)} rules")
        return enriched

    # ------------------------------------------------------------------
    # Graph augmentation
    # ------------------------------------------------------------------

    def _augment_with_graph(
        self, rules: List[Dict], query_date: date
    ) -> List[Dict]:
        """Augment rules with graph-based information (e.g., IPC→BNS mapping)."""
        for rule in rules:
            act_title = rule.get("act_title", "").upper()
            sec_num = str(rule.get("section_id", ""))

            # Check for act succession (IPC→BNS, CrPC→BNSS, etc.)
            successor = self.graph.find_successor_act(act_title)
            if successor and query_date.year >= successor["year"]:
                rule["successor_act"] = successor

            # Check for specific BNS section mapping
            if "PENAL CODE" in act_title and query_date.year >= 2024:
                bns_eq = self.graph.get_bns_equivalent(sec_num)
                if bns_eq:
                    rule["successor_section"] = bns_eq
                    logger.info(f"Augmented IPC {sec_num} with BNS {bns_eq.get('bns_section')}")

        return rules

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

    def _annotate_temporal_validity(self, rules: List[Dict], query_year: int):
        """Annotate each rule with whether it was temporally valid at query_year."""
        for r in rules:
            sy = r.get("start_year", 0)
            ey = r.get("end_year", 9999)
            r["temporally_valid"] = (sy <= query_year <= ey)

    # ------------------------------------------------------------------
    # Primary law detection
    # ------------------------------------------------------------------

    @staticmethod
    def get_primary_law(rules: List[Dict]) -> Optional[str]:
        """
        Determine the most relevant law from retrieval results.
        Uses frequency: the law_name appearing most often is the primary.
        """
        if not rules:
            return None
        from collections import Counter
        counts = Counter(r.get("law_name", "") for r in rules if r.get("law_name"))
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Timeline builder
    # ------------------------------------------------------------------

    def build_timeline_dict(
        self, rules: List[Dict], primary_law: Optional[str] = None,
    ) -> List[Dict]:
        """
        Build a timeline grouping rules by law_name+section.

        If primary_law is specified, only rules from that law are included
        to avoid cluttering the timeline with unrelated acts.
        """
        filtered = rules
        if primary_law:
            filtered = [r for r in rules if r.get("law_name", "") == primary_law]
            if not filtered:
                # Fall back to all rules if filter removed everything
                filtered = rules

        # Group rules by (law_name, section)
        groups: Dict[Tuple[str, str], List[Dict]] = {}
        for rule in filtered:
            key = (rule.get("law_name", ""), rule.get("section", ""))
            groups.setdefault(key, []).append(rule)

        timeline = []
        for (law, sec), group_rules in groups.items():
            # Deduplicate by rule_id and sort by start_year
            seen_ids: set = set()
            unique = []
            for r in group_rules:
                if r["rule_id"] not in seen_ids:
                    seen_ids.add(r["rule_id"])
                    unique.append(r)
            unique.sort(key=lambda r: r.get("start_year", 0))

            for rule in unique:
                start_year = rule.get("start_year", 0)
                end_year = rule.get("end_year", 9999)
                end_display = "Present" if end_year >= 9999 else str(end_year)

                text = rule.get("rule_text", "")
                preview = (text[:200] + "...") if len(text) > 200 else text
                preview = self._fix_text_encoding(preview)

                entry = {
                    "rule_id": rule.get("rule_id", ""),
                    "period": f"{start_year}-{end_display}",
                    "start_date": f"{start_year}-01-01",
                    "end_date": f"{end_year}-12-31" if end_year < 9999 else None,
                    "status": rule.get("status", "unknown"),
                    "version": None,
                    "amendment_ref": rule.get("successor_act", {}).get("new_act") if rule.get("successor_act") else None,
                    "rule_text_preview": preview,
                }
                timeline.append(entry)

        return timeline

    def build_version_timeline(self, law_name: str, section: str) -> List[Dict]:
        """
        Fetch all versions of a specific law+section from ChromaDB
        and return a structured version timeline.
        """
        results = self.chroma.query_timeline(law_name=law_name, section=section)
        if not results or not results.get("ids"):
            return []

        ids = results["ids"]
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        versions = []
        for i, rid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            start_year = meta.get("start_year", 0)
            end_year = meta.get("end_year", 9999)
            end_display = "Present" if end_year >= 9999 else str(end_year)

            text = docs[i] if i < len(docs) else ""
            preview = (text[:200] + "...") if len(text) > 200 else text
            preview = self._fix_text_encoding(preview)

            versions.append({
                "rule_id": rid,
                "period": f"{start_year}-{end_display}",
                "start_date": f"{start_year}-01-01",
                "end_date": f"{end_year}-12-31" if end_year < 9999 else None,
                "status": meta.get("status", "unknown"),
                "version": f"v{i + 1}",
                "amendment_ref": None,
                "rule_text_preview": preview,
            })

        return versions

    # ------------------------------------------------------------------
    # Encoding fix
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_text_encoding(text: str) -> str:
        """Fix mojibake in text from the dataset."""
        try:
            return text.encode('cp1252').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def calculate_confidence(
        self,
        rules: List[Dict],
        query_type: str,
        web_result: Optional[Dict] = None,
    ) -> float:
        """
        Calculate an evidence-based confidence score.

        Factors:
        - Number of temporally valid results
        - Semantic distance of top result
        - Whether web verification agrees
        - Query type specificity
        """
        if not rules:
            return 0.0

        # Base confidence from query type
        base = {
            "HISTORICAL_POINT": 0.80,
            "LATEST": 0.75,
            "DATE_RANGE": 0.78,
            "COMPARISON": 0.80,
            "GENERAL": 0.70,
        }.get(query_type, 0.70)

        # Boost for temporal validity
        valid_count = sum(1 for r in rules if r.get("temporally_valid"))
        if valid_count > 0:
            base += min(valid_count * 0.03, 0.10)

        # Boost/penalty from semantic distance (lower distance = better)
        top_dist = rules[0].get("distance", 1.0)
        if top_dist < 0.5:
            base += 0.08  # very close match
        elif top_dist < 1.0:
            base += 0.04
        elif top_dist > 2.0:
            base -= 0.08  # poor match

        # Boost from web verification agreement
        if web_result and web_result.get("verified"):
            web_conf = web_result.get("confidence", 0.5)
            base = (base + web_conf) / 2 + 0.05  # agreement bonus

        return round(min(max(base, 0.0), 0.99), 3)
