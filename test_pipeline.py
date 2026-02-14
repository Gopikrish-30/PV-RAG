"""End-to-end pipeline test for PV-RAG"""
import sys, os
os.environ.setdefault("GROQ_API_KEY", "test")

from datetime import date
from app.db.chromadb_manager import get_chroma_manager
from app.modules.query_parser import QueryParser
from app.modules.retrieval import TemporalRetrievalEngine
from app.modules.llm_response_generator import get_llm_generator

def main():
    # --- Test 1: ChromaDB connection ---
    chroma = get_chroma_manager()
    stats = chroma.get_stats()
    print(f"=== ChromaDB: {stats['total_rules']} documents, {stats['unique_acts_sample']} unique acts ===")

    # --- Test 2: Semantic search (no temporal filter) ---
    r = chroma.semantic_search("penalty for not wearing helmet on motorcycle", n_results=5)
    print("\n=== Semantic search: helmet penalty ===")
    for i in range(len(r["ids"][0])):
        rid = r["ids"][0][i]
        doc = r["documents"][0][i]
        meta = r["metadatas"][0][i]
        dist = r["distances"][0][i]
        print(f"  {i+1}. [{dist:.3f}] {meta.get('law_name','?')} - Sec {meta.get('section','?')} "
              f"({meta.get('start_year','?')}-{meta.get('end_year','?')}) | {doc[:80]}")

    # --- Test 3: Hybrid retrieval for year 2010 ---
    r2 = chroma.query_by_point_in_time("helmet penalty fine motorcycle", date(2010, 1, 1), n_results=5)
    print("\n=== Hybrid point-in-time (2010): helmet ===")
    for i in range(len(r2["ids"][0])):
        meta = r2["metadatas"][0][i]
        doc = r2["documents"][0][i]
        dist = r2["distances"][0][i]
        sy = meta.get("start_year", "?")
        ey = meta.get("end_year", "?")
        valid = "YES" if (isinstance(sy, int) and isinstance(ey, int) and sy <= 2010 <= ey) else "no"
        print(f"  {i+1}. [{dist:.3f}] {meta.get('law_name','?')} - Sec {meta.get('section','?')} "
              f"({sy}-{ey}) valid@2010={valid} | {doc[:80]}")

    # --- Test 4: Hybrid latest ---
    r3 = chroma.query_latest("helmet penalty fine motorcycle", n_results=5)
    print("\n=== Hybrid latest: helmet ===")
    for i in range(len(r3["ids"][0])):
        meta = r3["metadatas"][0][i]
        doc = r3["documents"][0][i]
        dist = r3["distances"][0][i]
        ey = meta.get("end_year", 0)
        active = "ACTIVE" if ey >= 9999 else f"ended {ey}"
        print(f"  {i+1}. [{dist:.3f}] {meta.get('law_name','?')} - Sec {meta.get('section','?')} "
              f"({meta.get('start_year','?')}-{ey}) {active} | {doc[:80]}")

    # --- Test 5: Full pipeline queries ---
    parser = QueryParser()
    engine = TemporalRetrievalEngine()
    gen = get_llm_generator()

    queries = [
        "What was the helmet fine in 2010?",
        "Current penalty for not wearing helmet",
        "Income tax section 80C",
        "What is the penalty for drunk driving?",
        "hi",
    ]

    for q in queries:
        print(f'\n{"="*60}')
        print(f'Pipeline: "{q}"')
        print("=" * 60)
        parsed = parser.parse(q)
        qt = parsed["query_type"]
        st = parsed.get("search_query", q)
        print(f"  Parsed: type={qt}, search=\"{st[:60]}\"")

        if qt == "HISTORICAL_POINT":
            qd = parsed["temporal_entities"].get("date", date.today())
            rules = engine.retrieve_for_point_in_time(st, qd, limit=5)
        elif qt == "LATEST":
            rules = engine.retrieve_latest(st, limit=5)
        else:
            rules = engine.retrieve_general(st, limit=5)

        print(f"  Results: {len(rules)} rules")
        for i, r in enumerate(rules[:3]):
            tv = r.get("temporally_valid", "?")
            print(f"    {i+1}. {r['law_name']} - Sec {r['section']} "
                  f"({r['start_year']}-{r['end_year']}) valid={tv} | {r['rule_text'][:60]}")

        # Generate response
        qd_obj = parsed["temporal_entities"].get("date") if qt == "HISTORICAL_POINT" else None
        answer = gen.generate_response(q, qt, rules, qd_obj)
        print(f"\n  Answer (first 300 chars):\n  {answer[:300]}")

    print("\n\n=== ALL TESTS COMPLETE ===")

if __name__ == "__main__":
    main()
