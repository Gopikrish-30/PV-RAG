"""
TLAG-RAG Step 1: Download all immediately-available sources.
Run this FIRST. Downloads everything that doesn't need scraping.
"""
import os, json, subprocess, urllib.request, zipfile, pathlib, sys, time

DATA_DIR = pathlib.Path(__file__).parent.parent
GITHUB_DIR = DATA_DIR / "github_acts"
ZENODO_DIR = DATA_DIR / "zenodo_acts"
LOG_FILE   = DATA_DIR / "logs" / "download_log.txt"

LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ─── 1. GITHUB REPOS ───────────────────────────────────────────────────────────

GITHUB_REPOS = [
    {
        "name":   "Indian-Law-Penal-Code-Json",
        "url":    "https://github.com/civictech-India/Indian-Law-Penal-Code-Json.git",
        "clone_to": str(GITHUB_DIR / "ipc_bns_crpc"),
        "desc":   "IPC, BNS, CrPC, CPC in JSON"
    },
    {
        "name":   "The_Constitution_Of_India",
        "url":    "https://github.com/Yash-Handa/The_Constitution_Of_India.git",
        "clone_to": str(GITHUB_DIR / "constitution"),
        "desc":   "Full Constitution JSON"
    },
    {
        "name":   "egazette",
        "url":    "https://github.com/sushant354/egazette.git",
        "clone_to": str(DATA_DIR.parent / "tools" / "egazette"),
        "desc":   "eGazette download tool"
    },
]

def clone_repos():
    for repo in GITHUB_REPOS:
        dest = pathlib.Path(repo["clone_to"])
        if dest.exists():
            log(f"  [SKIP] {repo['name']} already cloned")
            continue
        log(f"  [CLONE] {repo['name']} — {repo['desc']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth=1", repo["url"], str(dest)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            log(f"  [OK] {repo['name']}")
        else:
            log(f"  [FAIL] {repo['name']}: {result.stderr}")

# ─── 2. ZENODO ACTS DATASET ────────────────────────────────────────────────────

# Zenodo record for "Indian Central Acts" dataset (858 acts, 1838-2020)
# Multiple possible records — try each
ZENODO_RECORDS = [
    "7986743",   # Primary: 858 annotated Central Acts JSON
    "7550117",   # Alternative: IndiaLaw dataset
    "6519817",   # Alternative: Legal-NLP corpus
]

def download_zenodo():
    log("\n[ZENODO] Downloading 858 Central Acts dataset...")
    
    # Try zenodo_get tool first
    for record_id in ZENODO_RECORDS:
        log(f"  Trying record {record_id}...")
        result = subprocess.run(
            [sys.executable, "-m", "zenodo_get", record_id, "-o", str(ZENODO_DIR)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            log(f"  [OK] Downloaded record {record_id}")
            return
        else:
            log(f"  [FAIL] Record {record_id}: {result.stderr[:200]}")
    
    # Fallback: direct API download
    log("  Trying direct Zenodo API...")
    for record_id in ZENODO_RECORDS:
        try:
            api_url = f"https://zenodo.org/api/records/{record_id}"
            with urllib.request.urlopen(api_url, timeout=30) as resp:
                meta = json.loads(resp.read())
            
            for file_info in meta.get("files", []):
                fname = file_info["key"]
                furl  = file_info["links"]["self"]
                fpath = ZENODO_DIR / fname
                if fpath.exists():
                    log(f"  [SKIP] {fname}")
                    continue
                log(f"  [DOWNLOAD] {fname} ({file_info['size']/1e6:.1f} MB)")
                urllib.request.urlretrieve(furl, str(fpath))
                log(f"  [OK] {fname}")
            return
        except Exception as e:
            log(f"  [FAIL] Record {record_id}: {e}")
    
    log("  [WARNING] Zenodo download failed — will supplement from IndiaCode scraper")

# ─── 3. INSTALL REQUIREMENTS ───────────────────────────────────────────────────

def install_requirements():
    log("\n[SETUP] Installing required packages...")
    packages = [
        "requests", "beautifulsoup4", "lxml", "selenium",
        "pdfplumber", "PyPDF2", "pytesseract",
        "pandas", "tqdm", "groq",
        "zenodo-get", "networkx", "pyarrow",
        "aiohttp", "asyncio", "fake-useragent",
    ]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log("  [OK] All packages installed")
    else:
        log(f"  [PARTIAL] {result.stderr[:500]}")

# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 60)
    log("TLAG-RAG DATA COLLECTION — STEP 1: SOURCE DOWNLOADS")
    log("=" * 60)
    
    install_requirements()
    
    log("\n[GITHUB] Cloning structured JSON repositories...")
    clone_repos()
    
    download_zenodo()
    
    log("\n[DONE] Step 1 complete. Run 02_scrape_indiacode.py next.")
