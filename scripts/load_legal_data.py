"""
Load legal dataset CSV into ChromaDB
"""
import sys
import os
import pandas as pd
from datetime import datetime, date
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.chromadb_manager import get_chroma_manager
from config.settings import settings


def parse_year_to_date(year_value, is_start=True):
    """Convert year to date object"""
    if pd.isna(year_value) or year_value == '':
        return None
    
    try:
        year = int(float(year_value))
        if is_start:
            return date(year, 1, 1)  # Start of year
        else:
            return date(year, 12, 31)  # End of year
    except (ValueError, TypeError):
        return None


def generate_rule_topic(law_name, section):
    """Generate a rule topic from law name and section"""
    # Extract keywords from law name
    keywords = []
    
    # Common legal topics mapping
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
        if len(word) > 4 and word.upper() != 'THE' and word.upper() != 'ACT':
            return word.lower()
    
    return 'general'


def determine_status(start_date, end_date):
    """Determine rule status based on dates"""
    if end_date is None:
        return 'active'
    
    today = date.today()
    if end_date < today:
        return 'superseded'
    else:
        return 'active'


def load_csv_data(csv_path: str, batch_size: int = 100):
    """Load CSV data into ChromaDB"""
    logger.info(f"Loading dataset from: {csv_path}")
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return False
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")
        logger.info(f"CSV columns: {list(df.columns)}")
        
        # Initialize ChromaDB
        chroma = get_chroma_manager()
        
        # Clear existing data (optional - comment out to append)
        logger.warning("Clearing existing data...")
        chroma.clear_collection()
        
        # Statistics
        inserted = 0
        skipped = 0
        errors = 0
        
        # Process in batches
        for idx in range(0, len(df), batch_size):
            batch = df.iloc[idx:idx + batch_size]
            batch_records = []
            
            for _, row in batch.iterrows():
                try:
                    # Parse dates
                    start_date = parse_year_to_date(row.get('start_year'), is_start=True)
                    end_date = parse_year_to_date(row.get('end_year'), is_start=False)
                    
                    if start_date is None:
                        skipped += 1
                        continue
                    
                    # Extract rule_id
                    rule_id = str(row.get('rule_id', f"RULE_{idx}_{_}"))
                    
                    # Determine status
                    status = determine_status(start_date, end_date)
                    
                    # Generate topic
                    law_name = str(row.get('law_name', ''))
                    section = str(row.get('section', ''))
                    rule_topic = generate_rule_topic(law_name, section)
                    
                    # Create rule dictionary
                    rule_dict = {
                        'rule_id': rule_id,
                        'act_title': law_name,
                        'section_id': section,
                        'rule_topic': rule_topic,
                        'rule_text': str(row.get('rule_text', '')),
                        'start_date': start_date,
                        'end_date': end_date,
                        'source': str(row.get('source', '')) if pd.notna(row.get('source')) else '',
                        'source_url': str(row.get('india_code_url', '')) if pd.notna(row.get('india_code_url')) else '',
                        'status': status,
                        'metadata': {
                            'original_csv_index': int(idx + _),
                            'modified_law_details': str(row.get('modified_law_details', '')) if pd.notna(row.get('modified_law_details')) else '',
                        }
                    }
                    
                    batch_records.append(rule_dict)
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx + _}: {e}")
                    errors += 1
            
            # Bulk insert batch
            if batch_records:
                try:
                    chroma.add_rules_batch(batch_records)
                    inserted += len(batch_records)
                    logger.info(f"Progress: {inserted}/{len(df)} records")
                except Exception as e:
                    logger.error(f"Error inserting batch: {e}")
                    errors += len(batch_records)
        
        # Summary
        logger.info("=" * 60)
        logger.success(f"Data loading completed!")
        logger.info(f"  ✓ Inserted: {inserted}")
        logger.info(f"  - Skipped:  {skipped}")
        logger.info(f"  ✗ Errors:   {errors}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False


def verify_data_load():
    """Verify loaded data"""
    logger.info("Verifying loaded data...")
    
    try:
        chroma = get_chroma_manager()
        stats = chroma.get_stats()
        
        logger.info(f"Total rules in ChromaDB: {stats['total_rules']:,}")
        logger.info(f"Unique acts (sample): {stats.get('unique_acts_sample', 'N/A')}")
        logger.info(f"Active rules (sample): {stats.get('active_rules_sample', 'N/A')}")
        
        # Sample records
        logger.info("\nSample records:")
        sample_results = chroma.collection.get(limit=3, include=["metadatas", "documents"])
        
        for i, rule_id in enumerate(sample_results['ids']):
            meta = sample_results['metadatas'][i]
            doc_preview = sample_results['documents'][i][:100] + "..."
            logger.info(f"  - {rule_id}: {meta['act_title'][:50]}... ({meta['start_date']} to {meta.get('end_date', 'Present')})")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("PV-RAG Data Loading (ChromaDB)")
    logger.info("=" * 60)
    
    # Get CSV path
    csv_path = settings.dataset_path
    
    if not os.path.exists(csv_path):
        # Try current directory
        csv_path = "legal_dataset_extended_with_mods_20260205_210844.csv"
    
    # Load data
    if load_csv_data(csv_path):
        # Verify
        verify_data_load()
        
        logger.success("\n✓ Data loading completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start API: uvicorn app.main:app --reload --port 8000")
        logger.info("2. Visit: http://localhost:8000/docs")
    else:
        logger.error("\n✗ Data loading failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
