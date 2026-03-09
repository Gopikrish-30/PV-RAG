"""
TLAG-RAG Step 3: Scrape eGazette.nic.in for amendment notifications.
Extracts amendment events with exact effective dates and penalty amounts.
"""
import requests, re, json, time, pathlib, logging
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE     = "https://egazette.nic.in"
DATA_DIR = pathlib.Path(__file__).parent.parent / "gazette"
PDF_DIR  = DATA_DIR / "pdfs"
JSON_DIR = DATA_DIR / "json"
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(pathlib.Path(__file__).parent.parent / "logs" / "gazette.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("gazette")

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "TLAGResearch/1.0 (Academic)"

AMEND_PAT   = re.compile(r'section\s+(\d+[A-Z]?)\s+of\s+(.+?Act[^,\n]{0,60})\s+(?:shall\s+be\s+)?substituted', re.I)
EFF_DATE_PAT = re.compile(r'(?:w\.e\.f\.?|with\s+effect\s+from)\s+(?:the\s+)?(\d{1,2}[\.\-/]\d{1,2}[\.\-/]\d{4})', re.I)
PENALTY_PAT  = re.compile(r'(?:fine|rupees?|Rs\.?)\s*(?:of\s+)?(?:Rs\.?\s*)?(\d[\d,]+)', re.I)

def safe_get(url, delay=2):
    try:
        time.sleep(delay)
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        return resp
    except Exception as e:
        log.warning(f"GET FAIL {url}: {e}")
        return None

def parse_date(ds):
    if not ds:
        return None
    for fmt in ["%d.%m.%Y", "%d-%m-%Y", "%d/%m/%Y"]:
        try:
            from datetime import datetime
            return datetime.strptime(ds.strip(), fmt).strftime("%Y-%m-%d")
        except:
            pass
    y = re.search(r'\b(1[89]\d\d|20[012]\d)\b', ds or "")
    return y.group(1) if y else None

def get_gazette_entries_for_year(year):
    """Get gazette notification links for a given year."""
    entries = []
    # Try the central gazette listing page
    for url_pattern in [
        f"{BASE}/EGazettePortal/Gaz/Central/Central.aspx?strYear={year}",
        f"{BASE}/WriteReadData/GazettedDocument/{year}/",
        f"https://egazette.gov.in/Central/{year}/",
    ]:
        resp = safe_get(url_pattern, delay=1.5)
        if not resp:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            txt  = link.get_text(strip=True)
            if ".pdf" in href.lower() or "gazette" in href.lower():
                full_url = BASE + href if href.startswith("/") else href
                entries.append({
                    "year": year,
                    "subject": txt,
                    "pdf_url": full_url,
                    "date": None,
                    "number": txt
                })
        if entries:
            break
    return entries

def extract_events_from_text(text, meta):
    """Extract amendment events from text."""
    events = []
    eff_dates = [parse_date(m.group(1)) for m in EFF_DATE_PAT.finditer(text)]
    eff_date  = eff_dates[0] if eff_dates else meta.get("date")
    
    for m in AMEND_PAT.finditer(text):
        section, act_name = m.groups()
        context   = text[max(0, m.start()-50): m.start()+400]
        pen_match = PENALTY_PAT.search(context)
        events.append({
            "gazette_number":     meta.get("number",""),
            "gazette_date":       meta.get("date",""),
            "gazette_url":        meta.get("pdf_url",""),
            "year":               meta.get("year"),
            "target_act":         act_name.strip(),
            "target_section":     section.strip(),
            "amendment_type":     "substituted",
            "effective_date":     eff_date,
            "new_penalty_amount": pen_match.group(1).replace(",","") if pen_match else None,
            "raw_excerpt":        context[:250],
        })
    return events

def scrape_pdf_text(pdf_path):
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages[:20])
    except:
        pass
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            r = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in r.pages[:20])
    except:
        return ""

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year",   type=int, default=2026)
    p.add_argument("--pdf",        action="store_true")
    args = p.parse_args()

    log.info("STEP 3: eGazette Amendment Scraper")
    all_events = []

    for year in tqdm(range(args.start_year, args.end_year + 1), desc="Years"):
        yfile = JSON_DIR / f"gazette_{year}.json"
        if yfile.exists():
            with open(yfile) as f:
                all_events.extend(json.load(f))
            continue

        entries     = get_gazette_entries_for_year(year)
        year_events = []

        for entry in tqdm(entries[:300], desc=f"  {year}", leave=False):
            text = ""
            if args.pdf and entry.get("pdf_url"):
                pdf_url = entry["pdf_url"]
                fname   = re.sub(r'[^\w\.\-]', '_', pdf_url.split("/")[-1]) or f"gaz_{year}.pdf"
                fpath   = PDF_DIR / fname
                if not fpath.exists():
                    resp = safe_get(pdf_url, delay=0.5)
                    if resp:
                        fpath.write_bytes(resp.content)
                if fpath.exists():
                    text = scrape_pdf_text(str(fpath))

            if not text and entry.get("pdf_url"):
                resp = safe_get(entry["pdf_url"].replace(".pdf",""), delay=1)
                if resp:
                    text = BeautifulSoup(resp.text, "lxml").get_text(" ", strip=True)

            if text:
                year_events.extend(extract_events_from_text(text, entry))

        with open(yfile, "w", encoding="utf-8") as f:
            json.dump(year_events, f, ensure_ascii=False, indent=2)

        all_events.extend(year_events)
        log.info(f"Year {year}: {len(year_events)} amendment events")

    out = DATA_DIR / "all_gazette_amendments.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    log.info(f"Total amendment events: {len(all_events)}")
    log.info("Run 04_enrich_versions.py next.")
