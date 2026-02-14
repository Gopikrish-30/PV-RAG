"""
ChromaDB Manager - Vector Database for Legal Rules
Uses PersistentClient to connect to the stored ChromaDB data.
Field names match the data loader: law_name, section, rule_topic,
start_year, end_year, status, source, source_url, rule_id
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from datetime import date
from loguru import logger

from config.settings import settings


class ChromaDBManager:
    """Manage ChromaDB vector database for legal rules"""

    def __init__(self):
        """Initialize ChromaDB PersistentClient and collection"""
        logger.info(f"Initializing ChromaDB (PersistentClient) at: {settings.chroma_persist_dir}")

        # CRITICAL: Use PersistentClient so we can see the data already on disk
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"description": "Legal rules with temporal versioning"},
        )

        doc_count = self.collection.count()
        logger.info(f"ChromaDB collection '{settings.chroma_collection_name}' has {doc_count} documents")

        # Embedding model – must be the same one used when loading data
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info("ChromaDB Manager ready")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query_text: str,
        n_results: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> Dict:
        """Core semantic search with optional metadata filters."""
        query_embedding = self.embedding_model.encode(query_text).tolist()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.warning(f"Filtered query failed ({e}), retrying without filter")
            kwargs.pop("where", None)
            results = self.collection.query(**kwargs)

        return results

    def query_by_point_in_time(
        self,
        query_text: str,
        query_date: date,
        topic: Optional[str] = None,
        n_results: int = 10,
    ) -> Dict:
        """
        Hybrid retrieval for a specific point in time.

        Strategy:
          1. Pure semantic search (best relevance, no temporal filter)
          2. Temporal-filtered search (rules valid at query_year)
          3. Merge & deduplicate – temporal matches get a ranking boost
        """
        query_year = query_date.year

        # Pass 1: Pure semantic search – always finds the most relevant docs
        semantic_results = self.semantic_search(query_text, n_results)

        # Pass 2: Temporal-filtered search
        where_filter: Dict = {
            "$and": [
                {"start_year": {"$lte": query_year}},
                {"end_year": {"$gte": query_year}},
            ]
        }
        temporal_results = self.semantic_search(query_text, n_results, where_filter)

        merged = self._merge_results(semantic_results, temporal_results, n_results)
        return merged

    def query_latest(
        self,
        query_text: str,
        topic: Optional[str] = None,
        n_results: int = 10,
    ) -> Dict:
        """
        Hybrid retrieval for current/latest rules.

        Strategy:
          1. Pure semantic search (best relevance)
          2. Active-only search (end_year == 9999)
          3. Merge – active rules get a ranking boost
        """
        # Pass 1: Pure semantic search
        semantic_results = self.semantic_search(query_text, n_results)

        # Pass 2: Active-only
        where_filter = {"end_year": {"$eq": 9999}}
        active_results = self.semantic_search(query_text, n_results, where_filter)

        merged = self._merge_results(semantic_results, active_results, n_results)
        return merged

    def query_by_date_range(
        self,
        query_text: str,
        start_year: int,
        end_year: int,
        n_results: int = 20,
    ) -> Dict:
        """
        Hybrid retrieval for a year range.

        Strategy:
          1. Pure semantic search
          2. Year-range filtered search
          3. Merge
        """
        semantic_results = self.semantic_search(query_text, n_results)

        where_filter = {
            "$and": [
                {"start_year": {"$lte": end_year}},
                {"end_year": {"$gte": start_year}},
            ]
        }
        filtered_results = self.semantic_search(query_text, n_results, where_filter)

        merged = self._merge_results(semantic_results, filtered_results, n_results)
        return merged

    # ------------------------------------------------------------------
    # Merge helper
    # ------------------------------------------------------------------

    def _merge_results(
        self,
        primary: Dict,
        secondary: Dict,
        n_results: int = 10,
    ) -> Dict:
        """
        Merge two ChromaDB result sets by deduplication and combined scoring.
        Items appearing in *secondary* (the filtered/boosted set) get a
        relevance bonus so they rank higher, but items only in *primary*
        (pure semantic) are still kept.

        ChromaDB distances are L2 (lower = better).  We boost secondary
        matches by subtracting 0.15 from their distance.
        """
        BOOST = 0.15  # bonus applied to secondary matches

        seen: Dict[str, Dict] = {}  # id → {doc, meta, score}

        for results, is_boosted in [(primary, False), (secondary, True)]:
            ids_list = results.get("ids", [[]])
            ids = ids_list[0] if ids_list and isinstance(ids_list[0], list) else ids_list
            docs_list = results.get("documents", [[]])
            docs = docs_list[0] if docs_list and isinstance(docs_list[0], list) else docs_list
            metas_list = results.get("metadatas", [[]])
            metas = metas_list[0] if metas_list and isinstance(metas_list[0], list) else metas_list
            dists_list = results.get("distances", [[]])
            dists = dists_list[0] if dists_list and isinstance(dists_list[0], list) else dists_list

            for i in range(len(ids)):
                rid = ids[i]
                dist = dists[i] if i < len(dists) else 1.0
                score = dist - BOOST if is_boosted else dist

                if rid not in seen or score < seen[rid]["score"]:
                    seen[rid] = {
                        "id": rid,
                        "document": docs[i] if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                        "score": score,
                        "distance": dist,  # keep original distance too
                    }

        # Sort by combined score (lower = better)
        ranked = sorted(seen.values(), key=lambda x: x["score"])[:n_results]

        # Rebuild ChromaDB-style result dict
        merged: Dict = {
            "ids": [[r["id"] for r in ranked]],
            "documents": [[r["document"] for r in ranked]],
            "metadatas": [[r["metadata"] for r in ranked]],
            "distances": [[r["distance"] for r in ranked]],
        }
        logger.info(
            f"Merged results: {len(ranked)} unique docs "
            f"(primary={len(primary.get('ids', [[]])[0])}, "
            f"secondary={len(secondary.get('ids', [[]])[0])})"
        )
        return merged

    def query_timeline(self, law_name: str, section: str) -> Dict:
        """Get all versions of a specific rule sorted chronologically."""
        logger.info(f"Timeline for: {law_name} - {section}")

        results = self.collection.get(
            where={
                "$and": [
                    {"law_name": {"$eq": law_name}},
                    {"section": {"$eq": section}},
                ]
            },
            include=["documents", "metadatas"],
        )

        if results["ids"]:
            sorted_idx = sorted(
                range(len(results["metadatas"])),
                key=lambda i: results["metadatas"][i]["start_year"],
            )
            results["ids"] = [results["ids"][i] for i in sorted_idx]
            results["documents"] = [results["documents"][i] for i in sorted_idx]
            results["metadatas"] = [results["metadatas"][i] for i in sorted_idx]

        return results

    def get_stats(self) -> Dict:
        """Get database statistics."""
        count = self.collection.count()
        sample = self.collection.get(limit=min(count, 1000), include=["metadatas"])

        if sample["metadatas"]:
            unique_acts = len(set(m.get("law_name", "") for m in sample["metadatas"]))
            active_count = sum(1 for m in sample["metadatas"] if m.get("end_year") == 9999)
        else:
            unique_acts = 0
            active_count = 0

        return {
            "total_rules": count,
            "unique_acts_sample": unique_acts,
            "active_rules_sample": active_count,
        }

    # ------------------------------------------------------------------
    # Write helpers (used by data loaders)
    # ------------------------------------------------------------------

    def add_rules_batch(self, rules: List[Dict]):
        """Add multiple rules in batch."""
        if not rules:
            return

        ids, embeddings, documents, metadatas = [], [], [], []
        for rule in rules:
            embedding = self.embedding_model.encode(rule["rule_text"]).tolist()
            metadata = {
                "rule_id": rule["rule_id"],
                "law_name": rule.get("act_title", rule.get("law_name", ""))[:500],
                "section": rule.get("section_id", rule.get("section", ""))[:100],
                "rule_topic": rule.get("rule_topic", "general"),
                "start_year": (
                    rule["start_date"].year
                    if hasattr(rule.get("start_date"), "year")
                    else int(rule.get("start_year", 0))
                ),
                "end_year": (
                    rule["end_date"].year
                    if rule.get("end_date") and hasattr(rule["end_date"], "year")
                    else (int(rule["end_year"]) if rule.get("end_year") else 9999)
                ),
                "status": rule.get("status", "active"),
                "source": rule.get("source", "")[:500],
                "source_url": rule.get("source_url", "")[:500],
            }
            ids.append(rule["rule_id"])
            embeddings.append(embedding)
            documents.append(rule["rule_text"])
            metadatas.append(metadata)

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        logger.info(f"Batch added: {len(rules)} rules")

    def clear_collection(self):
        """Clear all data from collection."""
        logger.warning("Clearing collection...")
        self.client.delete_collection(settings.chroma_collection_name)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name
        )
        logger.info("Collection cleared")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_chroma_manager = None


def get_chroma_manager() -> ChromaDBManager:
    """Get or create ChromaDB manager singleton"""
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaDBManager()
    return _chroma_manager
