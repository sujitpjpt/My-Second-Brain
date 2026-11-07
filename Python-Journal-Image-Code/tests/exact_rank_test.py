"""Tests for unit, load_exact_index and exact_rank. Fake in-memory data, no journal data, no Ollama.

    python tests/exact_rank_test.py     # or: pytest tests/exact_rank_test.py
"""
import sys
from pathlib import Path

import chromadb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from eval_retrieval import exact_rank, load_exact_index, unit  # noqa: E402

# Row order = chunk-id order, as load_exact_index guarantees. Distances to QUERY: c0 0.0, c1 0.4, c2 0.4 (near-tie), c3 1.0
V = unit(np.array([[1.0, 0.0], [0.6, 0.8], [0.6 + 1e-9, 0.8], [0.0, 1.0]]))
METAS = [
    {"entry_id": "e0", "date": "2024-03-01"},
    {"entry_id": "e1", "date": "2024-03-05"},
    {"entry_id": "e2", "date": "2024-03-05"},
    {"entry_id": "e3", "date": "2024-04-02"},
]
QUERY = [1.0, 0.0]


def ids(metas):
    return [m["entry_id"] for m in metas]


def test_unit_gives_length_one():
    assert np.isclose(np.linalg.norm(unit(np.array([3.0, 4.0]))), 1.0)  # single vector
    assert np.allclose(np.linalg.norm(unit(np.array([[3.0, 4.0], [0.0, 2.0]])), axis=1), 1.0)  # matrix, row by row


def test_unit_keeps_direction():
    assert np.allclose(unit(np.array([3.0, 4.0])), [0.6, 0.8])


def test_load_exact_index_sorts_rows_by_id():
    col = chromadb.EphemeralClient().create_collection("exact_rank_test", metadata={"hnsw:space": "cosine"})
    col.add(  # added out of order on purpose
        ids=["c2", "c0", "c1"],
        embeddings=[[0.0, 5.0], [2.0, 0.0], [3.0, 4.0]],
        metadatas=[{"entry_id": "e2", "date": "d"}, {"entry_id": "e0", "date": "d"}, {"entry_id": "e1", "date": "d"}],
    )
    loaded, metas = load_exact_index(col)
    assert ids(metas) == ["e0", "e1", "e2"]  # metas follow the id sort...
    assert np.allclose(loaded, [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])  # ...and rows match them, at unit length


def test_exact_rank_is_best_first_with_distances():
    metas, dists = exact_rank(V, METAS, QUERY, n_results=4)
    assert ids(metas) == ["e0", "e1", "e2", "e3"]
    assert dists == [0.0, 0.4, 0.4, 1.0]


def test_exact_rank_near_tie_goes_to_lower_row():
    # c2 is closer by ~1e-9. Rounding makes it an exact tie, so the stable sort keeps c1 first.
    metas, _ = exact_rank(V, METAS, QUERY, n_results=3)
    assert ids(metas)[1:] == ["e1", "e2"]


def test_exact_rank_cuts_at_n_results():
    metas, dists = exact_rank(V, METAS, QUERY, n_results=2)
    assert ids(metas) == ["e0", "e1"]
    assert len(dists) == 2


def test_exact_rank_date_filter_excludes_other_dates():
    metas, _ = exact_rank(V, METAS, QUERY, n_results=4, dates=["2024-03-05", "2024-04-02"])
    assert ids(metas) == ["e1", "e2", "e3"]  # e0 is the closest chunk but outside the window


def test_exact_rank_date_filter_caps_results_and_never_returns_inf():
    metas, dists = exact_rank(V, METAS, QUERY, n_results=4, dates=["2024-03-01"])
    assert ids(metas) == ["e0"]  # asked for 4, only 1 chunk is on that date
    assert all(np.isfinite(dists))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
