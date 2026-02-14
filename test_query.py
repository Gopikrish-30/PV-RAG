import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import json

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path='./chroma_db',
    settings=ChromaSettings(anonymized_telemetry=False)
)
col = client.get_collection('legal_rules')

# Test semantic search WITHOUT temporal filter
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode('helmet fine motor vehicles penalty').tolist()

print("Testing semantic search without temporal filter...")
res = col.query(
    query_embeddings=[emb],
    n_results=5,
    include=['metadatas', 'documents']
)

print(f"\nFound {len(res['ids'][0])} results")

for i in range(min(3, len(res['ids'][0]))):
    print(f"\n{'='*60}")
    print(f"Result {i+1}:")
    print(json.dumps(res['metadatas'][0][i], indent=2))
    print(f"\nText preview: {res['documents'][0][i][:250]}...")
