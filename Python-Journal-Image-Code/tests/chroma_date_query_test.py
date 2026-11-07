"""Print the output structure of a date-filtered col.query(). Fake in-memory data, no journal data."""
from pprint import pprint

import chromadb

col = chromadb.EphemeralClient().create_collection("demo", metadata={"hnsw:space": "cosine"})
col.add(
    ids=["c0", "c1", "c2"],
    embeddings=[[1.0, 0.0], [0.5, 0.5], [0.9, 0.1]],
    metadatas=[
        {"entry_id": "2024-03-01_fake", "date": "2024-03-01"},
        {"entry_id": "2024-03-05_fake", "date": "2024-03-05"},
        {"entry_id": "2024-04-02_fake", "date": "2024-04-02"},  # outside the filter below
    ],
)

r = col.query(query_embeddings=[[1.0, 0.0]], n_results=2,
              where={"date": {"$in": ["2024-03-01", "2024-03-05"]}}, include=["metadatas", "distances"])
pprint(r)
