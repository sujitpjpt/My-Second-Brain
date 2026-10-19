"""Embed chunks_<version>.jsonl locally via Ollama -> ChromaDB collection (+ index manifest).

    python build_index.py --dry-run                  # verify hash, print stats, embed nothing
    python build_index.py                            # builds chroma_db/ collection nomic-embed-text_chunks_v1
    python build_index.py --rebuild                  # drop and recreate the collection
    python build_index.py --model bge-m3 --doc-prefix "" --query-prefix ""   # a variant to compare on the eval set

Everything stays on this machine: embeddings come from the local Ollama server, vectors go to the gitignored chroma_db/.
The vectors are journal content (they can be partly inverted to text), so treat chroma_db/ like the corpus.
Prints counts and dimensions only, never journal text.
"""
import argparse
import hashlib
import json
from pathlib import Path

import chromadb
import ollama

# Paths are relative to this file, so the script works from any working directory
REPO = Path(__file__).parent.parent
CORPUS = REPO / "Journal-Entires" / "corpus"
CHROMA_DIR = REPO / "chroma_db"  # gitignored


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_digest(model):
    """Digest of the local model, so the index records exactly which weights produced the vectors."""
    for m in ollama.list().models:
        if m.model in (model, f"{model}:latest"):
            return m.digest
    raise SystemExit(f"model {model!r} not found locally. Run: ollama pull {model}")


def main():
    ap = argparse.ArgumentParser()
    # Which frozen chunk file to index (chunks_v1.jsonl). A new chunking variant gets its own version and its own collection
    ap.add_argument("--version", default="v1", help="which chunks_<version>.jsonl to index")
    # Local Ollama embedding model. The name goes into the collection name, so a different model is a separate collection
    ap.add_argument("--model", default="nomic-embed-text")
    # nomic-embed-text is trained with task prefixes; omitting them noticeably hurts retrieval.
    # Pass "" for models that don't use prefixes (e.g. bge-m3)
    ap.add_argument("--doc-prefix", default="search_document: ", help="prepended to every chunk before embedding")
    # Not used here: recorded in the index manifest so the query/eval code applies the same prefix at search time
    ap.add_argument("--query-prefix", default="search_query: ", help="prepended to questions at search time")
    # Chunks sent to Ollama per embed call. 248 chunks is tiny, so this only sets how often progress prints
    ap.add_argument("--batch-size", type=int, default=32)
    # Without this, an existing collection is an error: an index is never silently overwritten
    ap.add_argument("--rebuild", action="store_true", help="drop the collection if it exists, then rebuild")
    # Verify the chunks hash and print stats, then exit without embedding or writing anything
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunks_path = CORPUS / f"chunks_{args.version}.jsonl"
    chunks_manifest = json.loads((CORPUS / f"chunks_manifest_{args.version}.json").read_text())
    # The index must be built from exactly the frozen chunks; a mismatch means the file changed under the manifest
    actual = sha256(chunks_path)
    if actual != chunks_manifest["sha256"]:
        raise SystemExit(f"{chunks_path.name} hash {actual[:12]} != manifest {chunks_manifest['sha256'][:12]}. Refusing to index.")

    chunks = [json.loads(ln) for ln in chunks_path.read_text().splitlines() if ln]
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id"
    print({"n_chunks": len(chunks), "chunks_sha256": actual[:12], "model": args.model})
    if args.dry_run:
        return

    digest = model_digest(args.model)
    # Model name in the collection name: a different model is a different collection, never a mixed one
    name = f"{args.model.replace(':', '-')}_chunks_{args.version}"
    manifest_path = CORPUS / f"index_manifest_{name}.json"

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if any(c.name == name for c in client.list_collections()):
        if not args.rebuild:
            raise SystemExit(f"collection {name} exists. Use --rebuild to replace it.")
        client.delete_collection(name)
        manifest_path.unlink(missing_ok=True)
    # Cosine distance: Chroma defaults to L2, but text-embedding similarity is conventionally cosine
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), args.batch_size):
        # Step through chunks in windows of batch_size: i = 0, B, 2B, ... ; slice i:i+B takes up to B chunks
        # (the last slice is shorter if len(chunks) isn't a multiple of B; Python slicing clamps at the end, no IndexError)
        batch = chunks[i:i + args.batch_size]
        # One local Ollama call per batch (not per chunk) -> one vector back per input, same order; nothing leaves the machine
        # input: doc_prefix + embed_text (embed_text includes the date header; stored `text` doesn't).
        #   doc_prefix is the task prefix the model was trained on for documents (e.g. "search_document: ");
        #   queries must use the matching query_prefix at search time, hence both go in the manifest
        # truncate=False: error out instead of silently cutting a chunk that exceeds the context window
        #   (a too-long chunk fails the whole batch, leaving a partial collection -> re-run with --rebuild)
        resp = ollama.embed(model=args.model, input=[args.doc_prefix + c["embed_text"] for c in batch], truncate=False)
        col.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=resp.embeddings,  # list of float vectors, aligned with ids/documents/metadatas below
            documents=[c["text"] for c in batch],  # what the generator receives, without the date header
            # Chroma metadata must be scalars, so the sources list is joined
            metadatas=[{
                "entry_id": c["entry_id"], "date": c["date"], "title": c["title"],
                "chunk_index": c["chunk_index"], "n_chunks": c["n_chunks"],
                "sources": ",".join(c["sources"]),
            } for c in batch],
        )
        print(f"embedded {min(i + args.batch_size, len(chunks))}/{len(chunks)}")

    assert col.count() == len(chunks), f"collection has {col.count()} vectors, expected {len(chunks)}"
    dim = len(col.get(ids=ids[:1], include=["embeddings"])["embeddings"][0])

    # Ties the index to its exact inputs and embedding setup; the eval harness reads query_prefix from here
    manifest = {
        "collection": name, "chroma_dir": CHROMA_DIR.name,
        "chunks_file": chunks_path.name, "chunks_sha256": actual,
        "model": args.model, "model_digest": digest, "dim": dim,
        "doc_prefix": args.doc_prefix, "query_prefix": args.query_prefix,
        "distance": "cosine", "n_vectors": col.count(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(0o444)
    print(f"indexed {col.count()} chunks (dim {dim}) into {name}; wrote {manifest_path.name}")


if __name__ == "__main__":
    main()
