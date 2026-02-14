"""
Temporal Retrieval Engine using ChromaDB (No PostgreSQL!)
"""
from typing import List, Dict, Optional
from datetime import date
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import settings


class TemporalRetrievalEngine:
    """Retrieve legal rules with temporal awareness using ChromaDB"""
    
    def __init__(self):
        """Initialize ChromaDB client and embedding model"""
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            self.collection = self.chroma_client.get_collection(name="legal_rules")
            logger.info(f"Connected to ChromaDB collection: {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Failed to get collection: {e}")
            raise
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
    
    def retrieve_for_point_in_time(
        self,
        query_text: str,
        query_date: date,
        topics: List[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Retrieve rules valid at a specific point in time
        
        ChromaDB filters on metadata:
        - start_year <= query_year
        - end_year >= query_year OR end_year == 9999 (active)
        """
        logger.info(f"Retrieving rules for date: {query_date}")
        
        query_year = query_date.year
        
        # Build where filter for temporal query
        where_filter = {
            "$and": [
                {"start_year": {"$lte": query_year}},
                {"end_year": {"$gte": query_year}}
            ]
        }
        
        # Add topic filter if provided
        if topics and topics != ['general']:
            where_filter["rule_topic"] = {"$in": topics}
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Query ChromaDB with semantic search + temporal filter
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            where=where_filter,
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = self._format_results(results)
        
        logger.info(f"Found {len(formatted_results)} rules valid at {query_date}")
        return formatted_results
    
    def retrieve_latest(
        self,
        query_text: str,
        topics: List[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Retrieve currently active rules
        """
        logger.info("Retrieving latest active rules")
        
        today = date.today()
        query_year = today.year
        
        # Build where filter for active rules
        where_filter = {
            "$and": [
                {"start_year": {"$lte": query_year}},
                {"end_year": {"$gte": query_year}},
                {"status": "active"}
            ]
        }
        
        # Add topic filter
        if topics and topics != ['general']:
            where_filter["rule_topic"] = {"$in": topics}
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            where=where_filter,
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = self._format_results(results)
        
        logger.info(f"Found {len(formatted_results)} active rules")
        return formatted_results
    
    def retrieve_by_date_range(
        self,
        query_text: str,
        start_year: int,
        end_year: int,
        topics: List[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Retrieve all rules that were active during a date range
        """
        logger.info(f"Retrieving rules for range: {start_year}-{end_year}")
        
        # Rules that overlap with the date range
        # Rule is valid if: start_year <= end_year AND end_year >= start_year
        where_filter = {
            "$and": [
                {"start_year": {"$lte": end_year}},
                {"end_year": {"$gte": start_year}}
            ]
        }
        
        if topics and topics != ['general']:
            where_filter["rule_topic"] = {"$in": topics}
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            where=where_filter,
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = self._format_results(results)
        
        logger.info(f"Found {len(formatted_results)} rules in range")
        return formatted_results
    
    def search_by_text(
        self,
        query_text: str,
        query_date: Optional[date] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Semantic search with optional temporal filter
        """
        logger.info(f"Text search: '{query_text[:50]}...'")
        
        # Build where filter
        where_filter = {}
        
        if query_date:
            query_year = query_date.year
            where_filter = {
                "$and": [
                    {"start_year": {"$lte": query_year}},
                    {"end_year": {"$gte": query_year}}
                ]
            }
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            where=where_filter if where_filter else None,
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = self._format_results(results)
        
        logger.info(f"Found {len(formatted_results)} matching rules")
        return formatted_results
    
    def retrieve_timeline(self, rule_id: str) -> List[Dict]:
        """
        Retrieve complete version timeline for a rule
        """
        logger.info(f"Building timeline for rule: {rule_id}")
        
        # Get the base rule first
        base_results = self.collection.get(
            ids=[rule_id],
            include=["metadatas"]
        )
        
        if not base_results['ids']:
            return []
        
        base_metadata = base_results['metadatas'][0]
        law_name = base_metadata['law_name']
        section = base_metadata['section']
        
        # Find all versions of same law and section
        where_filter = {
            "$and": [
                {"law_name": law_name},
                {"section": section}
            ]
        }
        
        # Get all matching documents
        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )
        
        # Format and sort by start_year
        timeline = []
        for i, metadata in enumerate(results['metadatas']):
            timeline.append({
                'rule_id': metadata['rule_id'],
                'start_year': metadata['start_year'],
                'end_year': metadata['end_year'] if metadata['end_year'] != 9999 else None,
                'status': metadata['status'],
                'document': results['documents'][i][:200] + '...' if len(results['documents'][i]) > 200 else results['documents'][i]
            })
        
        # Sort by start_year
        timeline.sort(key=lambda x: x['start_year'])
        
        logger.info(f"Found {len(timeline)} versions in timeline")
        return timeline
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into standard format"""
        formatted = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted
        
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            
            formatted.append({
                'rule_id': metadata['rule_id'],
                'law_name': metadata['law_name'],
                'section': metadata['section'],
                'rule_topic': metadata['rule_topic'],
                'start_year': metadata['start_year'],
                'end_year': metadata['end_year'] if metadata['end_year'] != 9999 else None,
                'status': metadata['status'],
                'source': metadata.get('source', ''),
                'source_url': metadata.get('source_url', ''),
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0.0
            })
        
        return formatted
    
    def build_timeline_dict(self, rules: List[Dict]) -> List[Dict]:
        """Convert list of rules to timeline format"""
        timeline = []
        
        for rule in sorted(rules, key=lambda r: r['start_year']):
            entry = {
                'rule_id': rule['rule_id'],
                'period': f"{rule['start_year']}-{rule['end_year'] if rule['end_year'] else 'Present'}",
                'start_date': f"{rule['start_year']}-01-01",
                'end_date': f"{rule['end_year']}-12-31" if rule['end_year'] else None,
                'status': rule['status'],
                'version': None,
                'amendment_ref': None,
                'rule_text_preview': rule['document'][:200] + '...' if len(rule['document']) > 200 else rule['document']
            }
            timeline.append(entry)
        
        return timeline


# Example usage
if __name__ == "__main__":
    engine = TemporalRetrievalEngine()
    
    # Test: Get rules for 2010
    print("\n=== Rules valid in 2010 ===")
    rules_2010 = engine.retrieve_for_point_in_time(
        query_text="helmet fine penalty",
        query_date=date(2010, 12, 31),
        topics=['traffic'],
        limit=5
    )
    
    for rule in rules_2010:
        print(f"{rule['rule_id']}: {rule['law_name'][:50]}... ({rule['start_year']} to {rule['end_year']})")
    
    # Test: Get latest rules
    print("\n=== Latest active rules ===")
    latest = engine.retrieve_latest(
        query_text="helmet fine",
        topics=['traffic'],
        limit=3
    )
    for rule in latest:
        print(f"{rule['rule_id']}: {rule['law_name'][:50]}... ({rule['start_year']} to {rule['end_year']})")
