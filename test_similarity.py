"""
Demonstrate how PV-RAG handles messy/misspelled/oddly-phrased queries.
Shows the full pipeline: query → embedding → similarity → results.
"""
from sentence_transformers import SentenceTransformer
from app.db.chromadb_manager import get_chroma_manager
from app.modules.query_parser import QueryParser
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
parser = QueryParser()
mgr = get_chroma_manager()

print(f"Total docs in ChromaDB: {mgr.collection.count()}")
print("=" * 80)

# ── Test queries: same intent, wildly different phrasing ──
queries = [
    "what i ridebike speed",                    # messy/broken English
    "What is the speed limit for riding a bike?",  # proper English
    "bike riding speed penalty",                 # keyword style
    "motor vehicle speed kya hai",               # Hinglish
    "helmet fine kaise bache",                   # Hinglish about helmet
    "drunk driving saza",                        # Hinglish about drunk driving
    "how much fine for overspeeding",            # common phrasing
    "punishment drink and drive",                # keyword jumble
]

for q in queries:
    print(f"\n{'─'*80}")
    print(f"USER QUERY: {q}")
    print(f"{'─'*80}")

    # Step 1: Query parser rewrites the query
    parsed = parser.parse(q)
    print(f"  Query Type : {parsed['query_type']}")
    print(f"  Topics     : {parsed['legal_topics']}")
    print(f"  Search Qry : {parsed['search_query']}")

    # Step 2: Embed the SEARCH QUERY (not the raw user query)
    search_text = parsed["search_query"]
    query_embedding = model.encode(search_text)

    # Step 3: Also embed the RAW user query to compare
    raw_embedding = model.encode(q)

    # Show how different the raw vs rewritten embeddings are
    cosine_sim = np.dot(query_embedding, raw_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(raw_embedding)
    )
    print(f"  Raw↔Rewritten embedding cosine similarity: {cosine_sim:.4f}")

    # Step 4: ChromaDB semantic search using the rewritten search query
    results = mgr.semantic_search(search_text, n_results=5)

    print(f"\n  Top 5 results (L2 distance — lower = more similar):")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        doc = results["documents"][0][i][:120].replace("\n", " ")
        print(f"    #{i+1}  dist={dist:.4f}  [{meta.get('law_name','')} / {meta.get('section','')}]")
        print(f"         {doc}...")

# ── Show WHY semantic search works with misspellings ──
print("\n" + "=" * 80)
print("HOW SEMANTIC SEARCH HANDLES MESSY QUERIES")
print("=" * 80)

# Embed several phrases and show their cosine similarity
pairs = [
    ("what i ridebike speed", "Motor Vehicles Act speed limit provision"),
    ("bike riding speed penalty", "Motor Vehicles Act speed limit provision"),
    ("drunk driving saza", "Driving by a drunken person or by a person under the influence of drugs"),
    ("helmet fine kaise bache", "Penalty for not wearing protective headgear"),
    ("ridebike speed", "speed limit motorcycle"),
]
for a, b in pairs:
    emb_a = model.encode(a)
    emb_b = model.encode(b)
    sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    print(f"  '{a}'\n    ↔ '{b}'\n    cosine similarity = {sim:.4f}\n")
