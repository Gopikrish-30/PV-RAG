"""
TLAG-RAG Step 5: Master Dataset Builder.
Merges all sources into one final temporally-versioned CSV + Parquet dataset.
Schema: Every ROW = one version of one section in one time period.
"""
import json, re, pathlib, logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime

DATA_DIR    = pathlib.Path(__file__).parent.parent
ENRICHED    = DATA_DIR / "enriched_acts"
RAW_DIR     = DATA_DIR / "raw_acts"
GITHUB_DIR  = DATA_DIR / "github_acts"
ZENODO_DIR  = DATA_DIR / "zenodo_acts"
FINAL_DIR   = DATA_DIR / "final_dataset"
FINAL_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(DATA_DIR / "logs" / "build_dataset.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("build")

PENALTY_PAT = re.compile(r'(?:fine|penalty|rupees?|Rs\.?)\s*(?:of\s+)?(?:Rs\.?\s*)?(\d[\d,]+)', re.I)
IMPRISON_PAT= re.compile(r'imprisonment\s+(?:for\s+)?(?:a\s+term\s+which\s+may\s+extend\s+to\s+)?(\d+)\s*(year|month)', re.I)

def parse_year(s):
    """Extract 4-digit year from a string."""
    if not s:
        return None
    if isinstance(s, (int, float)):
        return int(s) if 1800 < s < 2030 else None
    m = re.search(r'\b(1[89]\d\d|20[012]\d)\b', str(s))
    return int(m.group(1)) if m else None

def extract_penalty(text):
    fm = PENALTY_PAT.search(text or "")
    im = IMPRISON_PAT.search(text or "")
    return (
        fm.group(1).replace(",","") if fm else None,
        f"{im.group(1)} {im.group(2)}s" if im else None,
    )

def text_to_rows(act_data, source_label="IndiaCode"):
    """Convert one act JSON into flat rows for the dataset."""
    rows = []
    title       = act_data.get("title","")
    act_num     = act_data.get("act_number","")
    ministry    = act_data.get("ministry","")
    enact_date  = act_data.get("enactment_date","")
    source_url  = act_data.get("source_url","")
    category    = act_data.get("category","CentralAct")
    base_year   = parse_year(enact_date) or parse_year(title)

    for sec in act_data.get("sections", []):
        sec_id    = sec.get("section_id","")
        rule_text = sec.get("rule_text","")
        versions  = sec.get("versions", [])
        bns_map   = sec.get("ipc_bns_map")

        if not versions:
            # Single-version row with act base year
            fine, imp = extract_penalty(rule_text)
            rows.append({
                "provision_id":       f"{re.sub(r'[^\w]','_',title[:20])}_{sec_id}_{base_year}",
                "version_id":         f"{re.sub(r'[^\w]','_',title[:20])}_{sec_id}_{base_year}_v1",
                "law_name":           title,
                "act_number":         act_num,
                "ministry":           ministry,
                "category":           category,
                "section":            sec_id,
                "rule_text":          rule_text,
                "penalty_fine_inr":   fine,
                "penalty_imprisonment": imp,
                "has_penalty":        bool(fine or imp),
                "start_year":         base_year,
                "end_year":           None,
                "start_date":         enact_date or str(base_year),
                "end_date":           None,
                "status":             "active",
                "amendment_act":      None,
                "gazette_ref":        None,
                "effective_date":     None,
                "ipc_bns_mapping":    bns_map,
                "source_url":         source_url,
                "source_label":       source_label,
            })
        else:
            for vi, ver in enumerate(versions, 1):
                vtext = ver.get("version_text", rule_text)
                fine, imp = extract_penalty(vtext)
                start_yr  = parse_year(ver.get("effective_from"))
                end_yr    = parse_year(ver.get("effective_until"))
                rows.append({
                    "provision_id":       f"{re.sub(r'[^\w]','_',title[:20])}_{sec_id}_{base_year}",
                    "version_id":         f"{re.sub(r'[^\w]','_',title[:20])}_{sec_id}_{base_year}_v{vi}",
                    "law_name":           title,
                    "act_number":         act_num,
                    "ministry":           ministry,
                    "category":           category,
                    "section":            sec_id,
                    "rule_text":          vtext,
                    "penalty_fine_inr":   fine,
                    "penalty_imprisonment": imp,
                    "has_penalty":        bool(fine or imp),
                    "start_year":         start_yr or base_year,
                    "end_year":           end_yr,
                    "start_date":         ver.get("effective_from",""),
                    "end_date":           ver.get("effective_until",""),
                    "status":             ver.get("status","active"),
                    "amendment_act":      ver.get("amending_act",""),
                    "gazette_ref":        ver.get("gazette_ref",""),
                    "effective_date":     ver.get("effective_from",""),
                    "ipc_bns_mapping":    bns_map,
                    "source_url":         source_url,
                    "source_label":       source_label,
                })
    return rows

def load_github_acts():
    """Load structured JSON acts from GitHub repos."""
    rows = []
    for repo_dir in GITHUB_DIR.iterdir():
        if not repo_dir.is_dir():
            continue
        for jf in repo_dir.glob("**/*.json"):
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
                # Handle different JSON schemas
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "section" in item:
                            rows.append({
                                "provision_id":   f"GH_{jf.stem}_{item.get('section','')}",
                                "version_id":     f"GH_{jf.stem}_{item.get('section','')}_v1",
                                "law_name":       item.get("title", jf.stem.replace("_"," ")),
                                "act_number":     item.get("act_number",""),
                                "ministry":       "",
                                "category":       "CentralAct",
                                "section":        str(item.get("section","")),
                                "rule_text":      item.get("desc","") or item.get("text","") or item.get("content",""),
                                "penalty_fine_inr": None,
                                "penalty_imprisonment": None,
                                "has_penalty":    False,
                                "start_year":     parse_year(item.get("year","")),
                                "end_year":       None,
                                "start_date":     str(item.get("year","")),
                                "end_date":       None,
                                "status":         "active",
                                "amendment_act":  None,
                                "gazette_ref":    None,
                                "effective_date": None,
                                "ipc_bns_mapping": None,
                                "source_url":     "",
                                "source_label":   f"GitHub/{repo_dir.name}",
                            })
                elif isinstance(data, dict):
                    # Could be {"articles": [...]} style (Constitution)
                    items = data.get("articles") or data.get("sections") or data.get("rules") or []
                    for item in items:
                        if isinstance(item, dict):
                            rows.append({
                                "provision_id":   f"GH_{jf.stem}_{item.get('id','')}",
                                "version_id":     f"GH_{jf.stem}_{item.get('id','')}_v1",
                                "law_name":       data.get("title","") or jf.stem.replace("_"," "),
                                "act_number":     "",
                                "ministry":       "",
                                "category":       "Constitution" if "constitution" in jf.stem.lower() else "CentralAct",
                                "section":        str(item.get("id","") or item.get("article","")),
                                "rule_text":      item.get("desc","") or item.get("text","") or item.get("content",""),
                                "penalty_fine_inr": None,
                                "penalty_imprisonment": None,
                                "has_penalty":    False,
                                "start_year":     1950 if "constitution" in jf.stem.lower() else None,
                                "end_year":       None,
                                "start_date":     "1950-01-26" if "constitution" in jf.stem.lower() else "",
                                "end_date":       None,
                                "status":         "active",
                                "amendment_act":  None,
                                "gazette_ref":    None,
                                "effective_date": None,
                                "ipc_bns_mapping": None,
                                "source_url":     "",
                                "source_label":   f"GitHub/{repo_dir.name}",
                            })
            except Exception as e:
                log.debug(f"GitHub JSON skip {jf}: {e}")
    return rows

def load_existing_pv_rag():
    """Load the existing PV-RAG dataset as base layer."""
    existing = DATA_DIR.parent.parent / "legal_dataset_extended_with_mods_20260205_210844.csv"
    if not existing.exists():
        existing = list(DATA_DIR.parent.parent.glob("legal_dataset*.csv"))
        existing = existing[0] if existing else None
    if not existing:
        return []

    log.info(f"Loading existing PV-RAG dataset: {existing}")
    df = pd.read_csv(existing, low_memory=False)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "provision_id":   row.get("rule_id",""),
            "version_id":     str(row.get("rule_id","")) + "_v1",
            "law_name":       row.get("law_name",""),
            "act_number":     "",
            "ministry":       "",
            "category":       row.get("source",""),
            "section":        row.get("section",""),
            "rule_text":      str(row.get("rule_text","")),
            "penalty_fine_inr": None,
            "penalty_imprisonment": None,
            "has_penalty":    False,
            "start_year":     row.get("start_year"),
            "end_year":       row.get("end_year"),
            "start_date":     str(row.get("start_year","")),
            "end_date":       str(row.get("end_year","")) if row.get("end_year") else None,
            "status":         "active",
            "amendment_act":  None,
            "gazette_ref":    None,
            "effective_date": None,
            "ipc_bns_mapping": None,
            "source_url":     row.get("india_code_url",""),
            "source_label":   "PV-RAG-Original",
        })
    log.info(f"  Loaded {len(rows)} rows from existing dataset")
    return rows

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("STEP 5: Building Final Dataset")
    log.info("=" * 60)

    all_rows = []

    # 1. Load enriched acts from IndiaCode scraper
    act_dir = ENRICHED if any(ENRICHED.glob("*.json")) else RAW_DIR
    act_files = sorted(act_dir.glob("*.json"))
    log.info(f"Loading {len(act_files)} scraped acts from {act_dir.name}...")
    for af in tqdm(act_files, desc="Acts"):
        try:
            with open(af, encoding="utf-8") as f:
                data = json.load(f)
            all_rows.extend(text_to_rows(data))
        except Exception as e:
            log.warning(f"Skip {af.name}: {e}")

    # 2. Load GitHub acts
    log.info("Loading GitHub JSON acts...")
    gh_rows = load_github_acts()
    log.info(f"  {len(gh_rows)} rows from GitHub")
    all_rows.extend(gh_rows)

    # 3. Load original PV-RAG dataset
    pvrag_rows = load_existing_pv_rag()
    all_rows.extend(pvrag_rows)

    # 4. Load Zenodo acts if available
    for zf in ZENODO_DIR.glob("**/*.json"):
        try:
            with open(zf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "sections" in data:
                all_rows.extend(text_to_rows(data, source_label="Zenodo"))
        except Exception as e:
            log.debug(f"Zenodo skip {zf}: {e}")

    log.info(f"\nTotal rows before dedup: {len(all_rows)}")

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Extract penalty amounts for rows that have them
    mask = df["penalty_fine_inr"].isna() & df["rule_text"].notna()
    for idx in df[mask].index:
        fine, imp = extract_penalty(df.at[idx, "rule_text"])
        df.at[idx, "penalty_fine_inr"]      = fine
        df.at[idx, "penalty_imprisonment"]   = imp
        df.at[idx, "has_penalty"]            = bool(fine or imp)

    # Deduplicate by provision_id + version_id
    df = df.drop_duplicates(subset=["provision_id","version_id"])
    
    # Sort
    df = df.sort_values(["law_name","section","start_year"], na_position="first")

    log.info(f"Final rows: {len(df)}")
    log.info(f"Unique acts: {df['law_name'].nunique()}")
    log.info(f"Rows with penalty data: {df['has_penalty'].sum()}")
    log.info(f"Versioned rows (>1 version per section): { (df.groupby('provision_id').size() > 1).sum() }")

    # Save outputs
    csv_out     = FINAL_DIR / "tlag_rag_dataset_full.csv"
    parquet_out = FINAL_DIR / "tlag_rag_dataset_full.parquet"
    penalty_out = FINAL_DIR / "tlag_rag_penalty_sections.csv"
    stats_out   = FINAL_DIR / "dataset_stats.json"

    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    log.info(f"Saved CSV: {csv_out}")

    try:
        df.to_parquet(parquet_out, index=False)
        log.info(f"Saved Parquet: {parquet_out}")
    except:
        log.warning("pyarrow not available, skipping parquet")

    # Save penalty-only subset
    penalty_df = df[df["has_penalty"] == True].copy()
    penalty_df.to_csv(penalty_out, index=False, encoding="utf-8-sig")
    log.info(f"Saved penalty sections: {penalty_out} ({len(penalty_df)} rows)")

    # Save stats
    stats = {
        "total_rows":          len(df),
        "unique_acts":         int(df["law_name"].nunique()),
        "penalty_rows":        int(df["has_penalty"].sum()),
        "versioned_sections":  int((df.groupby("provision_id").size() > 1).sum()),
        "year_range":          [int(df["start_year"].min()), int(df["start_year"].max())] if df["start_year"].notna().any() else [None,None],
        "categories":          df["category"].value_counts().head(10).to_dict(),
        "built_at":            datetime.utcnow().isoformat() + "Z",
        "sources":             df["source_label"].value_counts().to_dict(),
    }
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    log.info("\n" + "="*50)
    log.info("DATASET BUILD COMPLETE")
    log.info("="*50)
    log.info(f"  Total rows:         {stats['total_rows']:,}")
    log.info(f"  Unique acts:        {stats['unique_acts']:,}")
    log.info(f"  Penalty sections:   {stats['penalty_rows']:,}")
    log.info(f"  Versioned sections: {stats['versioned_sections']:,}")
    log.info(f"  Year range:         {stats['year_range']}")
    log.info(f"\nFiles saved to: {FINAL_DIR}")
