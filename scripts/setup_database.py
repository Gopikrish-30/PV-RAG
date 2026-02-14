"""
Database setup and initialization script
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from app.db.session import Base, engine
from app.models.legal_models import LegalRule, WebVerificationLog, QueryLog
from config.settings import settings
from loguru import logger


def create_database():
    """Create database if it doesn't exist"""
    # Parse database URL to get connection details
    db_url = settings.database_url
    
    # Extract database name from URL
    # Format: postgresql://user:pass@host:port/dbname
    if '/' in db_url:
        db_name = db_url.split('/')[-1]
        base_url = db_url.rsplit('/', 1)[0]
        
        logger.info(f"Checking if database '{db_name}' exists...")
        
        try:
            # Connect to default 'postgres' database
            temp_engine = create_engine(f"{base_url}/postgres")
            with temp_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
                )
                exists = result.fetchone() is not None
                
                if not exists:
                    logger.info(f"Creating database '{db_name}'...")
                    # Create database (requires autocommit)
                    conn.execution_options(isolation_level="AUTOCOMMIT")
                    conn.execute(text(f"CREATE DATABASE {db_name}"))
                    logger.success(f"Database '{db_name}' created successfully!")
                else:
                    logger.info(f"Database '{db_name}' already exists.")
            
            temp_engine.dispose()
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            logger.warning("Please create the database manually or check credentials.")


def create_tables():
    """Create all database tables"""
    logger.info("Creating database tables...")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.success("All tables created successfully!")
        
        # Print created tables
        logger.info("Created tables:")
        for table in Base.metadata.sorted_tables:
            logger.info(f"  - {table.name}")
            
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


def create_indexes():
    """Create additional indexes for performance"""
    logger.info("Creating additional indexes...")
    
    indexes = [
        # Temporal query optimization
        "CREATE INDEX IF NOT EXISTS idx_temporal_lookup ON legal_rules(act_title, start_date, end_date)",
        
        # Topic search
        "CREATE INDEX IF NOT EXISTS idx_topic_temporal ON legal_rules(rule_topic, start_date, end_date)",
        
        # Version chain traversal
        "CREATE INDEX IF NOT EXISTS idx_version_chain ON legal_rules(previous_version_id)",
        
        # Web verification recent logs
        "CREATE INDEX IF NOT EXISTS idx_verification_recent ON web_verification_logs(rule_id, verification_date DESC)",
    ]
    
    try:
        with engine.connect() as conn:
            for idx_sql in indexes:
                conn.execute(text(idx_sql))
                conn.commit()
        
        logger.success(f"Created {len(indexes)} additional indexes")
    except Exception as e:
        logger.warning(f"Some indexes may already exist: {e}")


def verify_setup():
    """Verify database setup"""
    logger.info("Verifying database setup...")
    
    try:
        with engine.connect() as conn:
            # Check tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result]
            
            logger.info(f"Found {len(tables)} tables:")
            for table in tables:
                logger.info(f"  ✓ {table}")
            
            # Check if we can query
            result = conn.execute(text("SELECT COUNT(*) FROM legal_rules"))
            count = result.fetchone()[0]
            logger.info(f"Legal rules table has {count} records")
            
        logger.success("Database setup verified!")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("PV-RAG Database Setup")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create database
        create_database()
        
        # Step 2: Create tables
        create_tables()
        
        # Step 3: Create indexes
        create_indexes()
        
        # Step 4: Verify setup
        if verify_setup():
            logger.success("\n✓ Database setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Run: python scripts/load_legal_data.py")
            logger.info("2. Start API: uvicorn app.main:app --reload")
        else:
            logger.error("\n✗ Database setup verification failed!")
            
    except Exception as e:
        logger.error(f"\n✗ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
