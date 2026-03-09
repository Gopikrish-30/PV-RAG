"""
TLAG-RAG Step 4: LLM-Assisted Version Chain Enrichment.
Uses Groq to extract historical versions for complex sections.
Then merges with gazette data to build complete temporal chains.
"""
import json, re, pathlib, logging, os, time
from tqdm import tqdm

DATA_DIR   = pathlib.Path(__file__).parent.parent
RAW_DIR    = DATA_DIR / "raw_acts"
GAZ_FILE   = DATA_DIR / "gazette" / "all_gazette_amendments.json"
OUT_DIR    = DATA_DIR / "enriched_acts"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(DATA_DIR / "logs" / "enrich.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("enrich")

# Load environment
from dotenv import load_dotenv
load_dotenv(DATA_DIR.parent / ".env")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

# IPC to BNS Section Mapping (IPC 1860 → BNS 2023, effective 2024-07-01)
IPC_BNS_MAP = {
    "302":"103","303":"104","304":"105","304A":"106","304B":"80","305":"107",
    "306":"108","307":"109","308":"110","309":"226","310":"111","311":"112",
    "312":"88","313":"89","314":"90","315":"91","316":"92","317":"93",
    "318":"94","319":"95","320":"96","321":"116","322":"117","323":"115",
    "324":"118","325":"119","326":"120","326A":"121","326B":"122","327":"123",
    "328":"124","329":"125","330":"126","331":"127","332":"131","333":"132",
    "334":"133","335":"134","336":"125","337":"126","338":"127","339":"128",
    "340":"129","341":"135","342":"136","343":"137","344":"138","345":"139",
    "346":"140","347":"141","348":"142","349":"143","350":"144","351":"351",
    "354":"74","354A":"75","354B":"76","354C":"77","354D":"78","357":"143",
    "375":"63","376":"64","376A":"65","376AB":"66","376B":"67","376C":"68",
    "376D":"70","376DA":"71","376DB":"72","376E":"69","377":"38","378":"303",
    "379":"303","380":"305","381":"306","382":"307","384":"308","385":"309",
    "386":"310","387":"311","388":"312","389":"313","390":"308","391":"310",
    "392":"309","393":"310","394":"311","395":"310","396":"310","397":"312",
    "398":"311","399":"310","400":"311","401":"313","402":"312","403":"314",
    "404":"315","405":"316","406":"316","407":"317","408":"318","409":"319",
    "410":"317","411":"317","412":"318","413":"319","414":"319","415":"318",
    "416":"319","418":"319","419":"319","420":"318","421":"320","422":"321",
    "423":"321","424":"322","425":"324","426":"324","427":"324","428":"324",
    "429":"325","430":"326","431":"327","432":"328","433":"329","435":"330",
    "436":"331","437":"332","438":"333","440":"334","441":"329","442":"330",
    "443":"331","444":"332","445":"333","446":"334","447":"329","448":"330",
    "499":"356","500":"357","501":"358","502":"359","503":"351","504":"352",
    "505":"353","506":"351","507":"352","508":"353","509":"79",
}

def get_groq_client():
    if not GROQ_KEY:
        return None
    try:
        from groq import Groq
        return Groq(api_key=GROQ_KEY)
    except ImportError:
        log.warning("groq package not installed")
        return None

def llm_extract_versions(section_text, act_name, section_id, client):
    """Use Groq to extract historical versions from complex section text."""
    prompt = f"""You are a legal data extraction expert. Extract ALL historical versions of this Indian legal provision.

ACT: {act_name}
SECTION: {section_id}

SECTION TEXT (may contain footnote markers like 1[text]):
{section_text[:2000]}

Return a JSON object with key "versions" containing an array. Each version:
{{
  "version_text": "exact provision text for this version",
  "penalty_fine": "numeric amount in INR (digits only) or null",
  "penalty_imprisonment": "e.g. '3 years' or null",
  "effective_from": "YYYY-MM-DD or YYYY or null",
  "effective_until": "YYYY-MM-DD or YYYY or null (null = currently active)",
  "amending_act": "Name and number of amending act or null",
  "status": "active, superseded, or repealed"
}}

Rules:
- If no amendment history exists, return one version with status=active
- Focus on penalty amounts/fines that changed across amendments
- Only return the JSON, nothing else."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content.strip()
        # Extract JSON if wrapped in code block
        if "```" in content:
            content = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            content = content.group(1) if content else "{}"
        return json.loads(content).get("versions", [])
    except Exception as e:
        log.debug(f"LLM fail for {section_id}: {e}")
        return []

def simple_version_extract(section_text):
    """Regex-based version extraction as fallback."""
    versions = []
    # Match footnote patterns: 1[new text] with optional "for old text"
    FN_REF = re.compile(r'\d+\[([^\]]{1,300})\]')
    SUBST  = re.compile(
        r'(\d+)\.\s*Subs(?:tituted)?\s+by\s+(.+?)\s*\(w\.e\.f\.\s*([^)]+)\)',
        re.I
    )
    OLD_TEXT = re.compile(r'for\s+(?:the\s+words?|the\s+figures?|"([^"]{1,200})")', re.I)
    
    fn_map = {}
    for m in SUBST.finditer(section_text):
        num, act, date = m.groups()
        old_m = OLD_TEXT.search(section_text[m.start():m.start()+400])
        fn_map[num] = {
            "act": act.strip(), "date": date.strip(),
            "old": old_m.group(1) if old_m and old_m.group(1) else None
        }
    
    for m in FN_REF.finditer(section_text):
        txt = m.group(1)
        # Find which footnote number this is (look back for 1[...)
        pre = section_text[max(0, m.start()-5):m.start()]
        num_m = re.search(r'(\d+)$', pre)
        if num_m:
            fn_num = num_m.group(1)
            fn     = fn_map.get(fn_num, {})
            versions.append({
                "version_text":       txt.strip(),
                "effective_from":     fn.get("date"),
                "amending_act":       fn.get("act"),
                "status":             "active",
                "effective_until":    None,
            })
            if fn.get("old"):
                versions.append({
                    "version_text":   fn["old"],
                    "effective_until": fn.get("date"),
                    "status":         "superseded",
                    "effective_from":  None,
                    "amending_act":   None,
                })
    return versions

def load_gazette_events():
    """Load all gazette amendment events."""
    if not GAZ_FILE.exists():
        return {}
    with open(GAZ_FILE) as f:
        events = json.load(f)
    # Index by act + section
    index = {}
    for ev in events:
        key = (ev.get("target_act","").lower(), ev.get("target_section","").lower())
        if key[0]:
            index.setdefault(key, []).append(ev)
    return index

def enrich_act(act_file, gazette_idx, llm_client):
    """Enrich a single act JSON with version chains."""
    with open(act_file, encoding="utf-8") as f:
        act = json.load(f)
    
    title_lower = act["title"].lower()
    enriched_sections = []
    
    for sec in act.get("sections", []):
        sec_id   = sec.get("section_id","").lower()
        sec_text = sec.get("rule_text","")
        
        # Try to get versions
        versions = sec.get("versions", [])
        
        # If no versions found by scraper, try LLM
        if not versions and llm_client and sec.get("has_penalty") and len(sec_text) > 100:
            versions = llm_extract_versions(sec_text, act["title"], sec.get("section_id",""), llm_client)
            time.sleep(0.3)  # Rate limit: ~3 req/sec on free tier
        
        # If still no versions, try regex
        if not versions:
            versions = simple_version_extract(sec_text)
        
        # Merge gazette events for this section
        gaz_key     = (title_lower[:30], sec_id.replace("sec","").replace(".","").strip())
        gaz_events  = gazette_idx.get(gaz_key, [])
        for ev in gaz_events:
            if ev.get("new_penalty_amount"):
                versions.append({
                    "version_text":   f"Fine: ₹{ev['new_penalty_amount']}",
                    "effective_from": ev.get("effective_date"),
                    "amending_act":   ev.get("gazette_number",""),
                    "status":         "active",
                    "effective_until": None,
                    "source":         "gazette",
                })
        
        # Determine IPC→BNS mapping
        ipc_match = re.match(r'sec(?:tion)?\s*(\d+[A-Z]?)', sec_id, re.I)
        bns_map   = None
        if ipc_match and "penal code" in title_lower or "ipc" in title_lower:
            bns_map = IPC_BNS_MAP.get(ipc_match.group(1))
        
        enriched_sections.append({
            **sec,
            "versions":      versions,
            "version_count": len(versions),
            "ipc_bns_map":   bns_map,
        })
    
    act["sections"] = enriched_sections
    act["enriched"] = True
    
    out_file = OUT_DIR / act_file.name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(act, f, ensure_ascii=False, indent=2)
    
    return len(enriched_sections)

if __name__ == "__main__":
    log.info("STEP 4: LLM Version Enrichment")
    
    act_files    = sorted(RAW_DIR.glob("*.json"))
    gazette_idx  = load_gazette_events()
    llm_client   = get_groq_client()
    
    log.info(f"Acts to enrich: {len(act_files)}")
    log.info(f"Gazette events indexed: {sum(len(v) for v in gazette_idx.values())}")
    log.info(f"LLM: {'Groq connected' if llm_client else 'Disabled (no API key)'}")
    
    total_sections = 0
    for act_file in tqdm(act_files, desc="Enriching"):
        try:
            n = enrich_act(act_file, gazette_idx, llm_client)
            total_sections += n
        except Exception as e:
            log.error(f"FAIL {act_file.name}: {e}")
    
    log.info(f"Enriched {total_sections} sections across {len(act_files)} acts")
    log.info("Run 05_build_dataset.py next.")
