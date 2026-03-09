"""
TLAG-RAG: Master Pipeline Orchestrator
Runs all 5 steps end-to-end to build the complete Indian legal dataset.
"""
import subprocess, sys, pathlib, time, json
from datetime import datetime

SCRIPTS_DIR = pathlib.Path(__file__).parent / "scripts"
DATA_DIR    = pathlib.Path(__file__).parent
LOG_DIR     = DATA_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

def run_step(script_name, args="", desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Script: {script_name} {args}")
    print(f"  Time:   {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + (args.split() if args else [])
    log_file = LOG_DIR / f"{script_name}.stdout.log"
    
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(DATA_DIR)
        )
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        proc.wait()
    
    if proc.returncode != 0:
        print(f"  [WARNING] Step exited with code {proc.returncode} — continuing...")
    else:
        print(f"  [OK] Done at {datetime.now().strftime('%H:%M:%S')}")
    
    return proc.returncode

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",        type=int, default=0, help="Start from step N (1-5). 0=all")
    parser.add_argument("--scrape-limit",type=int, default=0, help="Limit IndiaCode to N acts (0=all)")
    parser.add_argument("--workers",     type=int, default=2, help="Parallel workers for scraper")
    parser.add_argument("--gazette-pdf", action="store_true",  help="Also download gazette PDFs")
    parser.add_argument("--gaz-start",   type=int, default=2000)
    parser.add_argument("--gaz-end",     type=int, default=2026)
    args = parser.parse_args()

    start_time = time.time()
    print(f"\nTLAG-RAG Full Data Collection Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    steps = [
        (1, "01_download_sources.py",  "",
            "STEP 1: Download sources (GitHub repos, Zenodo dataset, install packages)"),
        
        (2, "02_scrape_indiacode.py",
            f"--workers {args.workers}" + (f" --limit {args.scrape_limit}" if args.scrape_limit else ""),
            "STEP 2: Scrape all acts from IndiaCode.nic.in (may take hours)"),
        
        (3, "03_scrape_gazette.py",
            f"--start-year {args.gaz_start} --end-year {args.gaz_end}" + (" --pdf" if args.gazette_pdf else ""),
            "STEP 3: Scrape eGazette amendment events"),
        
        (4, "04_enrich_versions.py",  "",
            "STEP 4: LLM-assisted version chain enrichment"),
        
        (5, "05_build_dataset.py",    "",
            "STEP 5: Merge all sources into final dataset"),
    ]

    for step_num, script, script_args, desc in steps:
        if args.step and step_num < args.step:
            print(f"[SKIP] {desc}")
            continue
        run_step(script, script_args, desc)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed/3600:.1f} hours")
    print(f"Final dataset: {DATA_DIR}/final_dataset/tlag_rag_dataset_full.csv")
    
    # Print stats
    stats_file = DATA_DIR / "final_dataset" / "dataset_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"\nDataset Summary:")
        print(f"  Total rows:         {stats.get('total_rows',0):,}")
        print(f"  Unique acts:        {stats.get('unique_acts',0):,}")
        print(f"  Penalty sections:   {stats.get('penalty_rows',0):,}")
        print(f"  Versioned sections: {stats.get('versioned_sections',0):,}")
