"""
TLAG-RAG Step 2: Scrape ALL acts from IndiaCode.nic.in
Extracts: act title, sections, amendment history, footnotes, ministry, act number.
Respects rate limits (1 req/sec). Resumable — skips already-scraped acts.
"""
import requests, json, time, re, pathlib, logging, hashlib
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import urllib.parse

BASE_URL = "https://www.indiacode.nic.in"
DATA_DIR  = pathlib.Path(__file__).parent.parent / "raw_acts"
DATA_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(pathlib.Path(__file__).parent.parent / "logs" / "indiacode_scrape.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("indiacode")

HEADERS = {
    "User-Agent": "TLAG-RAG Legal Research Bot v1.0 (Academic Non-Commercial)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ─── FOOTNOTE PARSING ──────────────────────────────────────────────────────────

FOOTNOTE_REF    = re.compile(r'(\d+)\[([^\]]{0,500})\]')
FOOTNOTE_SUBST  = re.compile(
    r'(\d+)\.\s*Substituted\s+by\s+(.+?Act[^,\n]*(?:,\s*\d{4})?)[^(]*'
    r'\(w\.e\.f\.\s*([^)]+)\)',
    re.IGNORECASE
)
FOOTNOTE_INSERT = re.compile(
    r'(\d+)\.\s*Inserted\s+by\s+(.+?Act[^,\n]*(?:,\s*\d{4})?)[^(]*'
    r'\(w\.e\.f\.\s*([^)]+)\)',
    re.IGNORECASE
)
FOOTNOTE_OMIT   = re.compile(
    r'(\d+)\.\s*Omitted\s+by\s+(.+?Act[^,\n]*(?:,\s*\d{4})?)[^(]*'
    r'\(w\.e\.f\.\s*([^)]+)\)',
    re.IGNORECASE
)
PENALTY_AMT     = re.compile(
    r'(?:fine|penalty|rupees?|Rs\.?)\s*(?:of\s+)?(?:Rs\.?\s*)?(\d[\d,]+)',
    re.IGNORECASE
)
IMPRISONMENT_P  = re.compile(
    r'imprisonment\s+(?:of\s+)?(?:either\s+description\s+)?(?:for\s+)?'
    r'(?:a\s+term\s+)?(?:which\s+may\s+extend\s+to\s+)?(\d+)\s*(year|month)',
    re.IGNORECASE
)

def parse_penalty(text):
    """Extract structured penalty info from section text."""
    fine_match = PENALTY_AMT.search(text)
    imp_match  = IMPRISONMENT_P.search(text)
    return {
        "fine_amount":       fine_match.group(1).replace(",", "") if fine_match else None,
        "imprisonment":      f"{imp_match.group(1)} {imp_match.group(2)}s" if imp_match else None,
        "has_penalty":       bool(fine_match or imp_match),
    }

def parse_footnotes(text):
    """Parse all footnote-based amendment references."""
    amendments = {}
    for fn_type, pattern in [
        ("substituted", FOOTNOTE_SUBST),
        ("inserted",    FOOTNOTE_INSERT),
        ("omitted",     FOOTNOTE_OMIT),
    ]:
        for m in pattern.finditer(text):
            num, act, date = m.groups()
            amendments[num] = {
                "type":           fn_type,
                "amending_act":   act.strip(),
                "effective_date": date.strip(),
            }
    return amendments

def extract_inline_versions(section_text, footnote_map):
    """
    Match inline [n[text]] markers with footnote amendments
    to create version history entries.
    """
    versions = []
    for m in FOOTNOTE_REF.finditer(section_text):
        fn_num, current_text = m.groups()
        if fn_num in footnote_map:
            fn = footnote_map[fn_num]
            versions.append({
                "text":          current_text.strip(),
                "amending_act":  fn["amending_act"],
                "effective_date": fn["effective_date"],
                "type":          fn["type"],
            })
    return versions

# ─── INDIACODE SCRAPER ─────────────────────────────────────────────────────────

def safe_get(url, retries=3, delay=1.5):
    for attempt in range(retries):
        try:
            time.sleep(delay)
            resp = SESSION.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == retries - 1:
                log.warning(f"FAIL [{attempt+1}/{retries}] {url}: {e}")
                return None
            time.sleep(delay * (attempt + 1))
    return None

def get_all_act_handles():
    """Fetch paginated list of all Central Acts."""
    acts = []
    offset = 0
    rpp = 100
    col = "123456789/1362"  # Central Acts collection
    
    log.info("Fetching act list from IndiaCode...")
    
    while True:
        url = (f"{BASE_URL}/browse?type=Title&rpp={rpp}&etal=0"
               f"&offset={offset}&order=ASC&sort_by=2&col={col}")
        resp = safe_get(url)
        if not resp:
            break
        
        soup = BeautifulSoup(resp.text, "lxml")
        
        # Main act listing items
        items = soup.select("div.artifact-title a, td.titleColumn a")
        if not items:
            # Try alternative selectors
            items = soup.select("a[href*='/handle/']")
        
        if not items:
            log.info(f"No more acts at offset {offset}")
            break
        
        new_count = 0
        for item in items:
            href  = item.get("href", "")
            title = item.get_text(strip=True)
            if "/handle/" in href and title and len(title) > 3:
                acts.append({"title": title, "handle": href})
                new_count += 1
        
        if new_count == 0:
            break
        
        log.info(f"  Found {len(acts)} acts so far (offset={offset})")
        offset += rpp
    
    # Also fetch Repealed + Spent acts
    for category, cat_col in [
        ("RepeatedActs", "123456789/2345"),
        ("SpentActs",    "123456789/2346"),
    ]:
        offset = 0
        while True:
            url = (f"{BASE_URL}/browse?type=Title&rpp={rpp}&etal=0"
                   f"&offset={offset}&order=ASC&sort_by=2&col={cat_col}")
            resp = safe_get(url)
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            items = soup.select("div.artifact-title a, a[href*='/handle/']")
            if not items:
                break
            for item in items:
                href  = item.get("href", "")
                title = item.get_text(strip=True)
                if "/handle/" in href and title and len(title) > 3:
                    acts.append({"title": title, "handle": href, "category": category})
            offset += rpp
            if offset > 2000:  # Safety cap per category
                break
    
    # Deduplicate by handle
    seen = set()
    unique_acts = []
    for a in acts:
        key = a["handle"]
        if key not in seen:
            seen.add(key)
            unique_acts.append(a)
    
    log.info(f"Total unique acts found: {len(unique_acts)}")
    return unique_acts

def scrape_act(act_meta):
    """Scrape all sections + metadata for one act."""
    handle     = act_meta["handle"]
    title      = act_meta["title"]
    safe_name  = re.sub(r'[^\w\-]', '_', title[:60])
    out_file   = DATA_DIR / f"{safe_name}.json"
    
    if out_file.exists():
        return {"skipped": True, "title": title}
    
    act_url = f"{BASE_URL}{handle}?view_type=browser"
    resp = safe_get(act_url)
    if not resp:
        return {"error": "fetch_failed", "title": title}
    
    soup = BeautifulSoup(resp.text, "lxml")
    
    # ── Act-level metadata ──
    act_number  = ""
    enact_date  = ""
    ministry    = ""
    
    for row in soup.select("table.detailtable tr, .metadata-field"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            if "act number" in label or "number" in label:
                act_number = value
            elif "enactment" in label or "date" in label:
                enact_date = value
            elif "ministry" in label or "department" in label:
                ministry = value
    
    # Try DSpace metadata
    for meta in soup.select("meta[name], meta[property]"):
        nm = meta.get("name","") + meta.get("property","")
        content = meta.get("content","")
        if "dc.date" in nm.lower():
            enact_date = enact_date or content
        if "dc.publisher" in nm.lower() or "ministry" in nm.lower():
            ministry = ministry or content
    
    # ── Sections ──
    sections = []
    
    # Method 1: structured section items
    for sec_el in soup.select(".section-content, .akn-section, .section-body"):
        sec_id  = sec_el.get("data-id","") or sec_el.get("id","")
        sec_txt = sec_el.get_text(" ", strip=True)
        if sec_txt and len(sec_txt) > 20:
            sections.append({"section_id": sec_id, "text": sec_txt})
    
    # Method 2: table-based layout
    if not sections:
        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                sec_id  = cells[0].get_text(strip=True)
                sec_txt = cells[1].get_text(" ", strip=True)
                if sec_txt and len(sec_txt) > 30 and re.match(r'^(Sec|Art|Rule|Schedule|Chap)', sec_id, re.I):
                    sections.append({"section_id": sec_id, "text": sec_txt})
    
    # Method 3: grab full document text (fallback)
    if not sections:
        body = soup.select_one(".akn-body, .act-content, #act-content, main article")
        if body:
            full_text = body.get_text(" ", strip=True)
            # Split by Section markers
            parts = re.split(r'(?=(?:Section|Sec\.?|Article|Rule)\s+\d+[A-Z]?\.?\s)', full_text)
            for part in parts:
                if len(part) > 30:
                    m = re.match(r'((?:Section|Sec\.?|Article|Rule)\s+\d+[A-Z]?\.?)\s*(.*)', part, re.DOTALL)
                    if m:
                        sections.append({"section_id": m.group(1).strip(), "text": m.group(2).strip()})
                    else:
                        sections.append({"section_id": "PREAMBLE", "text": part.strip()})
    
    # ── Amendment history (from the act page) ──
    amendments_listed = []
    for amend_block in soup.select(".amendment-history li, .modification-history li, .amending-acts li"):
        amendments_listed.append(amend_block.get_text(strip=True))
    
    # Also scan full-page text for footnote definitions
    page_text    = soup.get_text(" ", strip=True)
    footnote_map = parse_footnotes(page_text)
    
    # ── Build structured sections with version info ──
    structured_sections = []
    for sec in sections:
        txt = sec["text"]
        
        # Extract versions from inline markers
        versions = extract_inline_versions(txt, footnote_map)
        
        # Extract penalty info
        penalty = parse_penalty(txt)
        
        # Determine year range (use act enact date as baseline)
        year_match = re.search(r'\b(1[89]\d\d|20[012]\d)\b', enact_date or title)
        base_year  = int(year_match.group(1)) if year_match else None
        
        structured_sections.append({
            "section_id":        sec["section_id"],
            "rule_text":         txt[:5000],  # cap at 5K chars
            "penalty_amount":    penalty["fine_amount"],
            "imprisonment":      penalty["imprisonment"],
            "has_penalty":       penalty["has_penalty"],
            "versions":          versions,
            "start_year":        base_year,
            "end_year":          None,  # None = still active, enriched later
            "status":            "active",
        })
    
    act_data = {
        "title":              title,
        "handle":             handle,
        "act_number":         act_number,
        "enactment_date":     enact_date,
        "ministry":           ministry,
        "category":           act_meta.get("category", "CentralAct"),
        "source_url":         f"{BASE_URL}{handle}",
        "amendments_listed":  amendments_listed,
        "sections":           structured_sections,
        "section_count":      len(structured_sections),
        "scraped_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(act_data, f, ensure_ascii=False, indent=2)
    
    return {"ok": True, "title": title, "sections": len(structured_sections)}

# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    parser.add_argument("--limit",   type=int, default=0,  help="0=all")
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("STEP 2: IndiaCode Full Scraper")
    log.info("=" * 60)
    
    # Save checkpoint of act list
    acts_list_file = DATA_DIR.parent / "logs" / "acts_list.json"
    if acts_list_file.exists():
        log.info("Loading cached act list...")
        with open(acts_list_file) as f:
            acts = json.load(f)
    else:
        acts = get_all_act_handles()
        with open(acts_list_file, "w") as f:
            json.dump(acts, f, indent=2)
    
    if args.limit:
        acts = acts[:args.limit]
    
    log.info(f"Scraping {len(acts)} acts with {args.workers} workers...")
    
    ok = 0; skip = 0; fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(scrape_act, a): a for a in acts}
        for fut in tqdm(futures, desc="Acts", unit="act"):
            result = fut.result()
            if result.get("ok"):
                ok += 1
            elif result.get("skipped"):
                skip += 1
            else:
                fail += 1
    
    log.info(f"\nDone: {ok} scraped, {skip} skipped, {fail} failed")
    log.info("Run 03_scrape_gazette.py next.")
