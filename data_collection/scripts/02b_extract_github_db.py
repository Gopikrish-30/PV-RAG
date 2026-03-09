"""
Extract IndiaLaw SQLite database into structured JSON per act.
Also extracts from IPC/MVA/CrPC JSON files with full section text.
"""
import sqlite3, json, re, pathlib

DB_PATH   = pathlib.Path("github_acts/ipc_bns_crpc/IndiaLaw.db")
JSON_DIR  = pathlib.Path("github_acts/ipc_bns_crpc")
RAW_DIR   = pathlib.Path("raw_acts")
RAW_DIR.mkdir(exist_ok=True)

# Act metadata
ACT_META = {
    "IPC":  {"title":"THE INDIAN PENAL CODE, 1860",  "year":1860, "num":"45 of 1860", "bns_year":2024},
    "NIA":  {"title":"THE NEGOTIABLE INSTRUMENTS ACT, 1881", "year":1881, "num":"26 of 1881"},
    "IEA":  {"title":"THE INDIAN EVIDENCE ACT, 1872", "year":1872, "num":"1 of 1872"},
    "CRPC": {"title":"THE CODE OF CRIMINAL PROCEDURE, 1973","year":1973, "num":"2 of 1974"},
    "HMA":  {"title":"THE HINDU MARRIAGE ACT, 1955",  "year":1955, "num":"25 of 1955"},
    "CPC":  {"title":"THE CODE OF CIVIL PROCEDURE, 1908","year":1908,"num":"5 of 1908"},
    "IDA":  {"title":"THE INDUSTRIAL DISPUTES ACT, 1947","year":1947,"num":"14 of 1947"},
    "MVA":  {"title":"THE MOTOR VEHICLES ACT, 1988",  "year":1988, "num":"59 of 1988"},
}

PENALTY_PAT = re.compile(r'(?:fine|rupees?|Rs\.?)\s*(?:of\s+)?(?:Rs\.?\s*)?(\d[\d,]+)', re.I)
IMPRISON_PAT= re.compile(r'imprisonment.*?(\d+)\s*(year|month)', re.I)

def extract_penalty(text):
    fm = PENALTY_PAT.search(text or "")
    im = IMPRISON_PAT.search(text or "")
    return (
        fm.group(1).replace(",","") if fm else None,
        f"{im.group(1)} {im.group(2)}s" if im else None,
    )

def extract_db():
    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in c.fetchall()]
    
    for table in tables:
        meta = ACT_META.get(table, {})
        title= meta.get("title", table)
        year = meta.get("year", 1900)
        
        # Get columns
        c.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in c.fetchall()]
        print(f"  {table}: columns={cols}")
        
        c.execute(f"SELECT * FROM {table}")
        rows = c.fetchall()
        
        sections = []
        for row in rows:
            row_dict = dict(zip(cols, row))
            
            # Find text column
            text = ""
            for col in ["description","text","content","desc","body","sectionText"]:
                if col in row_dict and row_dict[col]:
                    text = str(row_dict[col])
                    break
            if not text:
                # Use any string column > 50 chars
                for col in cols:
                    val = str(row_dict.get(col,""))
                    if len(val) > 50:
                        text = val
                        break
            
            # Find section ID
            sec_id = ""
            for col in ["section","sectionNo","sec","id","number","section_number"]:
                if col in row_dict and row_dict.get(col):
                    sec_id = str(row_dict[col])
                    break
            
            # Find title
            sec_title = ""
            for col in ["title","sectionTitle","name","heading","section_title"]:
                if col in row_dict and row_dict.get(col):
                    sec_title = str(row_dict[col])
                    break
            
            fine, imp = extract_penalty(text)
            
            sections.append({
                "section_id":   sec_id,
                "section_title":sec_title,
                "rule_text":    text,
                "penalty_fine_inr":  fine,
                "penalty_imprisonment": imp,
                "has_penalty":  bool(fine or imp),
                "start_year":   year,
                "end_year":     None,
                "status":       "active",
                "versions":     [],
                "raw_db_cols":  {k:str(v) for k,v in row_dict.items() if k not in ["description","text","content","body"]},
            })
        
        act_data = {
            "title":         title,
            "act_number":    meta.get("num",""),
            "enactment_date":str(year),
            "ministry":      "",
            "category":      "CentralAct",
            "source_url":    "https://github.com/civictech-India/Indian-Law-Penal-Code-Json",
            "amendments_listed": [],
            "sections":      sections,
            "section_count": len(sections),
            "source_label":  "GitHub-IndiaLaw-DB",
            "scraped_at":    "2026-02-22",
        }
        
        out_file = RAW_DIR / f"{table}_{year}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(act_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {table}: {len(sections)} sections → {out_file.name}")
    
    conn.close()

def extract_json_files():
    """Also extract MVA and other acts from their JSON files."""
    json_files = list(JSON_DIR.glob("*.json"))
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            
            act_key = jf.stem.upper()
            meta    = ACT_META.get(act_key, {})
            title   = meta.get("title", jf.stem.upper().replace("_"," "))
            year    = meta.get("year", 1900)
            
            sections = []
            items = []
            
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                for k in ["sections","acts","rules","articles","data"]:
                    if k in data and isinstance(data[k], list):
                        items = data[k]
                        break

            for item in items:
                if not isinstance(item, dict):
                    continue
                
                text = ""
                for k in ["description","desc","sectionDesc","text","content","body","section_desc"]:
                    if item.get(k):
                        text = str(item[k])
                        break
                
                sec_id = ""
                for k in ["sectionNo","section","no","id","number","section_number"]:
                    if item.get(k):
                        sec_id = str(item[k])
                        break
                
                sec_title = ""
                for k in ["title","sectionTitle","heading","name","section_title"]:
                    if item.get(k):
                        sec_title = str(item[k])
                        break
                
                fine, imp = extract_penalty(text)
                
                sections.append({
                    "section_id":        sec_id,
                    "section_title":     sec_title,
                    "rule_text":         text,
                    "penalty_fine_inr":  fine,
                    "penalty_imprisonment": imp,
                    "has_penalty":       bool(fine or imp),
                    "start_year":        year,
                    "end_year":          None,
                    "status":            "active",
                    "versions":          [],
                })
            
            if sections:
                act_data = {
                    "title":        title,
                    "act_number":   meta.get("num",""),
                    "enactment_date": str(year),
                    "ministry":     "",
                    "category":     "CentralAct",
                    "source_url":   f"https://github.com/civictech-India/Indian-Law-Penal-Code-Json/{jf.name}",
                    "sections":     sections,
                    "section_count":len(sections),
                    "source_label": "GitHub-JSON",
                    "scraped_at":   "2026-02-22",
                }
                out_file = RAW_DIR / f"GH_{jf.stem}_{year}.json"
                if not out_file.exists():
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(act_data, f, ensure_ascii=False, indent=2)
                    print(f"  JSON {jf.name}: {len(sections)} sections → {out_file.name}")
        except Exception as e:
            print(f"  SKIP {jf}: {e}")

def extract_constitution():
    """Extract Constitution of India JSON."""
    const_dir = pathlib.Path("github_acts/constitution")
    for jf in const_dir.glob("**/*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            
            articles = []
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                for k in ["articles","sections","parts","data"]:
                    if k in data and isinstance(data[k], list):
                        items = data[k]
                        break
                if not items:
                    items = [data]
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                text    = item.get("desc","") or item.get("text","") or item.get("content","")
                art_id  = item.get("article","") or item.get("id","") or item.get("number","")
                title   = item.get("title","") or item.get("heading","") or item.get("name","")
                articles.append({
                    "section_id":   f"Art.{art_id}",
                    "section_title":title,
                    "rule_text":    str(text),
                    "has_penalty":  False,
                    "start_year":   1950,
                    "end_year":     None,
                    "status":       "active",
                    "versions":     [],
                })
            
            if articles:
                act_data = {
                    "title": "THE CONSTITUTION OF INDIA, 1950",
                    "act_number": "Constitution of India",
                    "enactment_date": "1950-01-26",
                    "ministry": "Ministry of Law and Justice",
                    "category": "Constitution",
                    "source_url": "https://github.com/Yash-Handa/The_Constitution_Of_India",
                    "sections": articles,
                    "section_count": len(articles),
                    "source_label": "GitHub-Constitution",
                    "scraped_at": "2026-02-22",
                }
                out_file = RAW_DIR / "CONSTITUTION_1950.json"
                if not out_file.exists():
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(act_data, f, ensure_ascii=False, indent=2)
                    print(f"  Constitution: {len(articles)} articles saved")
                break
        except Exception as e:
            print(f"  Constitution SKIP {jf}: {e}")

if __name__ == "__main__":
    print("Extracting GitHub IndiaLaw.db...")
    extract_db()
    print("\nExtracting GitHub JSON files...")
    extract_json_files()
    print("\nExtracting Constitution JSON...")
    extract_constitution()
    
    print("\nAll GitHub sources extracted to raw_acts/")
    print(f"Files: {len(list(RAW_DIR.glob('*.json')))}")
