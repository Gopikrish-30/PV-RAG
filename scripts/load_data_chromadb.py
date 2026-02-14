"""
Load legal dataset into ChromaDB (No PostgreSQL needed!)
"""
import sys
import os
import pandas as pd
from datetime import date
from loguru import logger
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings


def parse_year(year_value):
    """Convert year to integer"""
    if pd.isna(year_value) or year_value == '':
        return None
    try:
        return int(float(year_value))
    except (ValueError, TypeError):
        return None


def generate_rule_topic(law_name):
    """Generate a rule topic from law name"""
    topic_map = {
        'motor': 'traffic',
        'vehicles': 'traffic',
        'helmet': 'helmet_fine',
        'copyright': 'copyright',
        'patent': 'patent',
        'trademark': 'trademark',
        'tax': 'taxation',
        'income': 'income_tax',
        'contract': 'contracts',
        'employment': 'labor',
        'labour': 'labor',
        'consumer': 'consumer_protection',
        'environment': 'environment',
        'data': 'data_protection',
        'privacy': 'privacy',
    }
    
    law_lower = law_name.lower()
    for key, value in topic_map.items():
        if key in law_lower:
            return value
    
    # Default: extract first significant word
    words = law_name.split()
    for word in words:
        if len(word) > 4 and word.upper() not in ['THE', 'ACT']:
            return word.lower()
    
    return 'general'


def load_csv_to_chromadb(csv_path: str, batch_size: int = 100):
    """Load CSV data into ChromaDB"""
    logger.info(f"Loading dataset from: {csv_path}")
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return False
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        collection_name = "legal_rules"
        
        # Reset collection if it exists (for fresh load)
        try:
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Indian Legal Rules with Temporal Versioning"}
        )
        logger.info(f"Created collection: {collection_name}")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Statistics
        inserted = 0
        skipped = 0
        errors = 0
        
        # Process in batches
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            # Prepare batch data
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in batch_df.iterrows():
                try:
                    # Parse dates
                    start_year = parse_year(row.get('start_year'))
                    end_year = parse_year(row.get('end_year'))
                    
                    if start_year is None:
                        skipped += 1
                        continue
                    
                    # Create searchable document text
                    law_name = str(row.get('law_name', ''))
                    section = str(row.get('section', ''))
                    rule_text = str(row.get('rule_text', ''))
                    
                    # Combine for embedding (limit length)
                    document = f"{law_name} {section} {rule_text}"[:2000]
                    
                    # Generate rule_id
                    rule_id = str(row.get('rule_id', f"RULE_{idx}"))
                    
                    # Create metadata (ChromaDB supports filtering on these!)
                    metadata = {
                        'rule_id': rule_id,
                        'law_name': law_name[:500],
                        'section': section[:100],
                        'rule_topic': generate_rule_topic(law_name),
                        'start_year': start_year,
                        'end_year': end_year if end_year else 9999,  # Use 9999 for NULL/active
                        'status': 'active' if end_year is None else 'superseded',
                        'source': str(row.get('source', ''))[:500] if pd.notna(row.get('source')) else '',
                        'source_url': str(row.get('india_code_url', ''))[:500] if pd.notna(row.get('india_code_url')) else '',
                    }
                    
                    documents.append(document)
                    metadatas.append(metadata)
                    ids.append(rule_id)
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    errors += 1
            
            # Generate embeddings for batch
            if documents:
                try:
                    logger.info(f"Processing batch {batch_start}-{batch_end}...")
                    embeddings = embedding_model.encode(documents, show_progress_bar=False)
                    
                    # Add to ChromaDB
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings.tolist()
                    )
                    
                    inserted += len(documents)
                    logger.info(f"Inserted: {inserted}/{len(df)}")
                    
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    errors += len(documents)
        
        # Summary
        logger.info("=" * 60)
        logger.success(f"Data loading completed!")
        logger.info(f"  ✓ Inserted: {inserted}")
        logger.info(f"  - Skipped:  {skipped}")
        logger.info(f"  ✗ Errors:   {errors}")
        logger.info("=" * 60)
        
        # Verify
        count = collection.count()
        logger.info(f"Total documents in ChromaDB: {count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("PV-RAG Data Loading (ChromaDB)")
    logger.info("=" * 60)
    
    # Get CSV path
    csv_path = settings.dataset_path
    
    if not os.path.exists(csv_path):
        csv_path = "legal_dataset_extended_with_mods_20260205_210844.csv"
    
    # Load data
    if load_csv_to_chromadb(csv_path):
        logger.success("\n✓ Data loading completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start API: uvicorn app.main:app --reload")
        logger.info("2. Visit: http://localhost:8000/docs")
    else:
        logger.error("\n✗ Data loading failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
