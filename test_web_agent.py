"""
Test the Agentic Web Verification system end-to-end.
Checks: Tavily connectivity, search, decision logic, LLM compare, full pipeline.
"""
import traceback
from datetime import date, datetime

print("=" * 80)
print("AGENTIC WEB VERIFICATION — FULL DIAGNOSTIC")
print("=" * 80)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Check dependencies & config
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1] DEPENDENCIES & CONFIG")

from config.settings import settings
print(f"  Tavily API key set    : {bool(settings.tavily_api_key)}")
print(f"  Groq API key set      : {bool(settings.groq_api_key)}")
print(f"  web_verification_enabled: {settings.web_verification_enabled}")
print(f"  max_search_results    : {settings.max_search_results}")

try:
    from tavily import TavilyClient
    print(f"  tavily-python import  : OK")
except ImportError:
    print(f"  tavily-python import  : MISSING — pip install tavily-python")

try:
    from langchain_groq import ChatGroq
    print(f"  langchain-groq import : OK")
except ImportError:
    print(f"  langchain-groq import : MISSING")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Initialize the agent
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2] AGENT INITIALIZATION")

from app.agents.web_agent import WebVerificationAgent, get_web_agent
agent = get_web_agent()

print(f"  Tavily client ready   : {agent.client is not None}")
print(f"  agent.enabled         : {agent.enabled}")
print(f"  Groq LLM ready        : {agent.llm is not None}")

if not agent.enabled:
    print("\n  ⚠ Tavily client failed — checking API key validity...")
    try:
        test_client = TavilyClient(api_key=settings.tavily_api_key)
        r = test_client.search("test", max_results=1)
        print(f"  Tavily direct test    : OK ({len(r.get('results', []))} results)")
        # Fix the agent
        agent.client = test_client
        print(f"  Agent patched         : agent.enabled = {agent.enabled}")
    except Exception as e:
        print(f"  Tavily direct test    : FAILED — {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Tavily search test
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3] TAVILY SEARCH TEST")

if agent.enabled:
    try:
        results = agent.client.search(
            query="Motor Vehicles Act 2019 amendment penalty India",
            search_depth="advanced",
            max_results=3,
        )
        n = len(results.get("results", []))
        print(f"  Search returned       : {n} results")
        for i, r in enumerate(results.get("results", [])[:3]):
            print(f"    #{i+1} [{r.get('score', 0):.3f}] {r.get('title', '')[:70]}")
            print(f"        URL: {r.get('url', '')[:80]}")
    except Exception as e:
        print(f"  Search FAILED: {e}")
        traceback.print_exc()
else:
    print("  SKIPPED — Tavily not enabled")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Decision logic tests
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4] DECISION LOGIC (judge_need_for_web_verification)")

sample_rules = [
    {"law_name": "THE MOTOR VEHICLES ACT, 1988", "section": "Sec185",
     "start_year": 2019, "end_year": 9999, "status": "active",
     "rule_text": "Driving by a drunken person..."},
    {"law_name": "THE MOTOR VEHICLES ACT, 1988", "section": "Sec183",
     "start_year": 2019, "end_year": 9999, "status": "active",
     "rule_text": "Driving at excessive speed..."},
]
stale_rules = [
    {"law_name": "OLD ACT", "section": "Sec1",
     "start_year": 2000, "end_year": 2010, "status": "repealed",
     "rule_text": "Some old provision..."},
]

test_cases = [
    ("What was the fine in 2010?", "HISTORICAL_POINT", sample_rules, False,
     "Historical → should NOT verify"),
    ("What is the current penalty for drunk driving?", "LATEST", sample_rules, True,
     "Latest → should verify"),
    ("Tell me about IPC sections", "GENERAL", sample_rules, False,
     "General with active rules → may not verify"),
    ("What is the current fine?", "GENERAL", stale_rules, True,
     "General with stale rules → should verify"),
    ("What is the latest law?", "GENERAL", [], True,
     "No rules found → should verify"),
]

for query, qtype, rules, expected, desc in test_cases:
    result = agent.judge_need_for_web_verification(query, qtype, rules)
    status = "✓" if result == expected else "✗"
    print(f"  {status} {desc}")
    print(f"      query='{query[:50]}', type={qtype}, result={result}, expected={expected}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Build search query test
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5] SEARCH QUERY CONSTRUCTION")

test_primaries = [
    {"law_name": "THE MOTOR VEHICLES ACT, 1988", "section": "Sec185"},
    {"law_name": "THE INCOME TAX ACT, 1961", "section": "80C"},
    {},
]
for p in test_primaries:
    sq = agent._build_search_query("What is the penalty", p)
    print(f"  primary={p.get('law_name','(none)')[:40]}")
    print(f"    → '{sq}'")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Full verify_and_compare test
# ──────────────────────────────────────────────────────────────────────────────
print("\n[6] FULL verify_and_compare() TEST")

if agent.enabled:
    test_queries = [
        ("What is the current penalty for drunk driving in India?", sample_rules),
        ("Latest helmet fine Motor Vehicles Act", sample_rules),
    ]
    for query, rules in test_queries:
        print(f"\n  Query: {query}")
        result = agent.verify_and_compare(query, rules)
        print(f"    verified     : {result['verified']}")
        print(f"    confidence   : {result['confidence']:.2f}")
        print(f"    sources      : {len(result['sources'])} URLs")
        for s in result['sources'][:3]:
            print(f"      - {s[:80]}")
        web_ans = result['web_answer'][:200].replace('\n', ' ')
        print(f"    web_answer   : {web_ans}...")
        combined = result['combined_answer'][:200].replace('\n', ' ')
        print(f"    combined_ans : {combined}...")
else:
    print("  SKIPPED — Tavily not enabled")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Integration with main pipeline
# ──────────────────────────────────────────────────────────────────────────────
print("\n[7] MAIN PIPELINE INTEGRATION (LATEST query triggers web)")

from app.modules.query_parser import QueryParser
from app.db.chromadb_manager import get_chroma_manager
from app.modules.retrieval import TemporalRetrievalEngine

parser = QueryParser()
mgr = get_chroma_manager()
retrieval = TemporalRetrievalEngine()

# A "LATEST" query should auto-trigger web verification
query = "What is the current penalty for not wearing helmet?"
parsed = parser.parse(query)
print(f"  Query       : {query}")
print(f"  Type        : {parsed['query_type']}")
print(f"  needs_web   : {parsed.get('needs_web_verification', False)}")

# Retrieve from ChromaDB
rules = retrieval.retrieve_latest(parsed["search_query"])
print(f"  DB results  : {len(rules)} rules")
if rules:
    r = rules[0]
    print(f"  Top result  : {r.get('law_name','')} / {r.get('section','')}")

# Decision
need = agent.judge_need_for_web_verification(query, parsed["query_type"], rules)
print(f"  Need web?   : {need}")

if need and agent.enabled:
    web = agent.verify_and_compare(query, rules)
    print(f"  Web verified: {web['verified']}, confidence={web['confidence']:.2f}")
    print(f"  Sources     : {web['sources'][:2]}")
elif not agent.enabled:
    print(f"  Web agent not enabled — would skip verification")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Confidence scoring test
# ──────────────────────────────────────────────────────────────────────────────
print("\n[8] CONFIDENCE SCORING ANALYSIS")

# Simulate different source scenarios
mock_results_gov = {
    "results": [
        {"score": 0.9, "url": "https://indiacode.nic.in/...", "title": "T1", "content": "C1"},
        {"score": 0.85, "url": "https://legislative.gov.in/...", "title": "T2", "content": "C2"},
        {"score": 0.7, "url": "https://livelaw.in/...", "title": "T3", "content": "C3"},
    ]
}
mock_results_blog = {
    "results": [
        {"score": 0.5, "url": "https://randomlawblog.com/...", "title": "T1", "content": "C1"},
    ]
}

gov_sources = [r["url"] for r in mock_results_gov["results"]]
blog_sources = [r["url"] for r in mock_results_blog["results"]]

gov_conf = agent._calculate_confidence(mock_results_gov, gov_sources)
blog_conf = agent._calculate_confidence(mock_results_blog, blog_sources)

print(f"  3 gov.in sources (high score)  → confidence = {gov_conf:.2f}")
print(f"  1 random blog (low score)      → confidence = {blog_conf:.2f}")
print(f"  Gov.in gets higher confidence  : {'✓' if gov_conf > blog_conf else '✗'}")

print("\n" + "=" * 80)
print("WEB VERIFICATION DIAGNOSTIC COMPLETE")
print("=" * 80)
