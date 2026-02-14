"""
Quick verification script to test PV-RAG installation
"""
import sys
import os

print("=" * 70)
print("PV-RAG Installation Verification")
print("=" * 70)

errors = []
warnings = []
success = []

# Test 1: Check Python version
print("\n[1/8] Checking Python version...")
if sys.version_info >= (3, 10):
    success.append("Python version OK: " + sys.version.split()[0])
else:
    errors.append(f"Python 3.10+ required, found: {sys.version}")

# Test 2: Check required files
print("[2/8] Checking required files...")
required_files = [
    'requirements.txt',
    'app/main.py',
    'app/modules/query_parser.py',
    'app/modules/retrieval.py',
    'app/models/legal_models.py',
    'scripts/setup_database.py',
    'scripts/load_legal_data.py',
    'config/settings.py'
]

for file in required_files:
    if os.path.exists(file):
        success.append(f"✓ {file}")
    else:
        errors.append(f"✗ Missing file: {file}")

# Test 3: Check CSV dataset
print("[3/8] Checking dataset...")
dataset_file = 'legal_dataset_extended_with_mods_20260205_210844.csv'
if os.path.exists(dataset_file):
    size_mb = os.path.getsize(dataset_file) / (1024 * 1024)
    success.append(f"✓ Dataset found ({size_mb:.1f} MB)")
else:
    errors.append(f"✗ Dataset not found: {dataset_file}")

# Test 4: Check .env file
print("[4/8] Checking environment configuration...")
if os.path.exists('.env'):
    success.append("✓ .env file exists")
else:
    warnings.append("⚠ .env file not found (copy from .env.example)")

# Test 5: Try importing dependencies
print("[5/8] Checking Python dependencies...")
try:
    import fastapi
    import sqlalchemy
    import pydantic
    import pandas
    success.append("✓ Core dependencies installed")
except ImportError as e:
    errors.append(f"✗ Missing dependencies: {e}")
    warnings.append("Run: pip install -r requirements.txt")

# Test 6: Check database connection (if .env exists)
print("[6/8] Checking database connection...")
if os.path.exists('.env'):
    try:
        sys.path.insert(0, os.path.abspath('.'))
        from config.settings import settings
        from sqlalchemy import create_engine
        
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            success.append("✓ Database connection successful")
    except Exception as e:
        warnings.append(f"⚠ Database connection failed: {str(e)[:50]}...")
        warnings.append("  → Run: python scripts/setup_database.py")
else:
    warnings.append("⚠ Skipping database test (no .env file)")

# Test 7: Check if tables exist
print("[7/8] Checking database tables...")
if os.path.exists('.env'):
    try:
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if 'legal_rules' in tables:
            success.append(f"✓ Database tables exist ({len(tables)} tables)")
        else:
            warnings.append("⚠ Database tables not created")
            warnings.append("  → Run: python scripts/setup_database.py")
    except:
        warnings.append("⚠ Could not check tables")

# Test 8: Check if data is loaded
print("[8/8] Checking loaded data...")
if os.path.exists('.env'):
    try:
        from app.db.session import SessionLocal
        from app.models.legal_models import LegalRule
        
        db = SessionLocal()
        count = db.query(LegalRule).count()
        db.close()
        
        if count > 0:
            success.append(f"✓ Data loaded: {count:,} records")
        else:
            warnings.append("⚠ No data in database")
            warnings.append("  → Run: python scripts/load_legal_data.py")
    except Exception as e:
        warnings.append(f"⚠ Could not check data: {str(e)[:50]}...")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

if success:
    print(f"\n✓ Success ({len(success)}):")
    for item in success[:10]:  # Show first 10
        print(f"  {item}")
    if len(success) > 10:
        print(f"  ... and {len(success) - 10} more")

if warnings:
    print(f"\n⚠ Warnings ({len(warnings)}):")
    for item in warnings:
        print(f"  {item}")

if errors:
    print(f"\n✗ Errors ({len(errors)}):")
    for item in errors:
        print(f"  {item}")

# Final verdict
print("\n" + "=" * 70)
if errors:
    print("❌ Installation has ERRORS - please fix the issues above")
    sys.exit(1)
elif warnings:
    print("⚠️  Installation INCOMPLETE - follow the warnings above")
    print("\nQuick fix:")
    print("  1. Create .env: copy .env.example .env")
    print("  2. Edit .env with your database credentials")
    print("  3. Run: python scripts/setup_database.py")
    print("  4. Run: python scripts/load_legal_data.py")
    sys.exit(0)
else:
    print("✅ Installation COMPLETE - Ready to run!")
    print("\nStart the server:")
    print("  uvicorn app.main:app --reload --port 8000")
    print("\nThen visit: http://localhost:8000/docs")
    sys.exit(0)
