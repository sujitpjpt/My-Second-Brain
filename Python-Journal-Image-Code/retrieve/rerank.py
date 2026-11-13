"""Rerank the top dense-search hits with a local cross-encoder (default BAAI/bge-reranker-v2-m3).

    python retrieve/rerank.py --download          # one time: fetch weights, print the commit hash to pin below
    from retrieve.rerank import load_reranker, rerank
    model, info = load_reranker()                 # offline, pinned revision, MPS if available
    reordered = rerank(model, question, hits, depth=40)

Dense search embeds question and chunk separately, so a chunk is compressed into one vector before the question is
known. A cross-encoder reads (question, chunk) together and scores the pair, which is far more accurate but costs one
model pass per pair, so it only reorders the top `depth` candidates that dense search already found.

Privacy: inference is fully local. HF_HUB_OFFLINE=1 is forced before any Hugging Face import, so an eval run cannot
reach the network; only `--download` goes online, and it sends nothing but the model name. Nothing here prints text.
"""
import argparse
import os

# Must be set before huggingface_hub / sentence_transformers are imported (they read these once, at import time). The library reads these values once, when it's first imported. Setting them afterwards does nothing. So they must be set before huggingface_hub or sentence_transformers is imported.

# By default, loading a model makes a quick request to huggingface.co, even when the weights are already cached, to check whether a newer version exists. With this set to 1:
# No HTTP calls are made. The model loads only from ~/.cache/huggingface/hub.
# If the weights aren't cached, it fails with an error instead of quietly downloading.

# Forced, not setdefault: an exported HF_HUB_OFFLINE=0 in the shell must not silently put eval runs online.

os.environ["HF_HUB_OFFLINE"] = "1"

# When the library does go online, it normally sends anonymous usage data: library version, which model and so on. This turns that off. With offline mode on, nothing is sent anyway, so this is a backup in case offline mode is ever switched off (for example during --download).

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
# Pinned weights, so a run next month scores with the same model. Filled from the --download output.
# A model missing here is refused rather than loaded at whatever revision happens to be cached.
PINNED_REVISIONS = {
    # "BAAI/bge-reranker-v2-m3": "<commit hash printed by --download>",
}
# Token budget per (question, chunk) pair. Longest chunk is ~2000 chars (chunks_manifest_v1), roughly 500-800 tokens
# with the XLM-R tokenizer; 1024 leaves room for the question. Pairs over budget are counted, never silently cut.
MAX_LENGTH = 1024
BATCH_SIZE = 16


def load_reranker(model_name=DEFAULT_MODEL, device=None):
    """Load the pinned cross-encoder from the local cache. Returns (model, info); info goes into result files."""
    # If the model has no pinned hash, it stops with instructions instead of loading an unknown version.
    if model_name not in PINNED_REVISIONS:
        raise SystemExit(f"{model_name} has no pinned revision: run `python retrieve/rerank.py --download "
                         f"--model {model_name}` and add the printed hash to PINNED_REVISIONS")
    import torch  # heavy imports deferred so `import retrieve.rerank` stays cheap when --rerank is off
    from sentence_transformers import CrossEncoder
    # MPS = Metal Performance Shaders, Apple's way for PyTorch to run on the M-series GPU.
    # It returns True on  Mac, as long as your PyTorch build supports it, which modern builds do.
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    # fp32 (the default): half precision on MPS adds score noise that can swap near-tied candidates between runs
    # each weight is stored using 32 bits (4 bytes) of memory.
    # identical runs matter for us
    model = CrossEncoder(model_name, revision=PINNED_REVISIONS[model_name], device=device, max_length=MAX_LENGTH)
    # A PyTorch model has two modes: 1. train - Training the model,Randomly switches off some neurons on each pass ("dropout") 2. eval - Using the model,Uses every neuron, every time; no randomness
    model.model.eval()  # inference mode: no dropout, so scores are a pure function of the input
    info = {"rerank_model": model_name, "rerank_revision": PINNED_REVISIONS[model_name],
            "rerank_device": device, "rerank_max_length": MAX_LENGTH}
    return model, info


def n_over_budget(model, question, texts):
    """How many (question, chunk) pairs exceed MAX_LENGTH tokens and would be truncated. A count, never the text.

    Only a check: it does not change scores or ranking.

    texts is a list of chunks (all chunks being reranked, up to 40), not a single chunk.

    Example (MAX_LENGTH = 1024, texts = [A, B, C]):
        question = 7 tokens, chunk A = 300, chunk B = 1050, chunk C = 700
        each pair also gets 4 special tokens
        pair lengths:  A = 311,  B = 1061,  C = 711
        over 1024?     no        yes        no
        returns 1  -> chunk B is too long, its last 37 tokens are cut off
    """
    # Input: two lists, paired by position (not joined together):
    #   [q, q, q] and [A, B, C]  ->  (q, A), (q, B), (q, C)
    # Each pair is tokenized on its own, same as calling the tokenizer once per pair in a loop.
    # truncation=False: keep full lengths, otherwise nothing could ever be over the limit
    enc = model.tokenizer([question] * len(texts), texts, truncation=False)
    # Output: enc["input_ids"] is one list of token ids per pair (no padding, so lengths differ):
    #   [[0, 4865, ..., 2],    <- pair A, 311 ids
    #    [0, 4865, ..., 2],    <- pair B, 1061 ids
    #    [0, 4865, ..., 2]]    <- pair C, 711 ids
    # Count pairs longer than the limit: [False, True, False] -> 1
    return sum(len(ids) > MAX_LENGTH for ids in enc["input_ids"])


def rerank(model, question, hits, depth):
    """Reorder the first `depth` hits by cross-encoder score; hits past `depth` keep their dense order after them.

    hits: best-first list of (meta, dense_dist, text), as dense search returned them.
    Returns (reordered, scores, n_truncated): reordered is the same tuples in the new order; scores[i] is the
    rerank score of reordered[i] (None for hits past `depth`); n_truncated counts over-budget pairs.

    Worked example, depth=3, dense order A B C D:
        scores A 0.2 | B 1.7 | C 1.7 | D (not scored)
        -> B C A D. B and C tie after rounding, so B (better dense rank) stays first; D is appended unscored.
    """
    # The hits passed is still in dense order for one particular question. Example:
    # hits = [
    # (meta_A, 0.31, "text A..."),   # position 0
    # (meta_B, 0.35, "text B..."),   # position 1
    # (meta_C, 0.40, "text C..."),   # position 2
    # (meta_D, 0.42, "text D..."),   # position 3
    # (meta_E, 0.47, "text E..."),   # position 4
    # ]
    # depth = 3

    head, tail = hits[:depth], hits[depth:]
    # _ is a Python convention for "a value I'm required to name but don't need". The tuple has 3 parts, so 3 names are needed for unpacking. Only the text is kept.
    texts = [t for _, _, t in head]
    # Raw logits (identity activation), not sigmoid: sigmoid squashes confident scores toward 1.0, and rounding
    # those to 6 decimals would create false ties at the top of the list
    import torch
    # [(question, t) for t in texts] builds one pair per chunk, question first: [(question, "text A..."), (question, "text B..."), (question, "text C...")]
    
    # batch_size=BATCH_SIZE (16): how many pairs go to the GPU at once. With 40 pairs that's 3 passes (16 + 16 + 8). It affects speed and memory only, not the scores.
    
    # The model's final output is a raw number called a logit. It can be any value: −8.3, 0.2, 5.8, 15.0. By default, predict() passes this through a sigmoid function, 1 / (1 + e^(−x)), which converts any number into a value between 0 and 1:
    # Logits 15 and 16 are clearly different, but after sigmoid and rounding both become 1.0, which is a false tie. High logits are exactly the strongest matches, the chunks competing for rank 1, so this is the worst place to lose the difference.
    # torch.nn.Identity() is a function that returns its input unchanged. Passing it as the activation means "don't apply sigmoid; give me the raw logits". Then 15.0 and 16.0 stay 15.0 and 16.0.
    # Ranking only needs order, not values between 0 and 1. Sigmoid always keeps the order, so skipping it loses nothing.

    # model.predict(...) runs each pair through the model and returns one number per pair, in the same order as the input:
    # raw = array([ 0.2000003, 5.8124519, 5.8124517 ])   # a numpy array
    #             pair A     pair B     pair C
    # Higher means more relevant. A large positive number means the model is confident the chunk answers the question.
    # Lower means less relevant. A large negative number means the model is confident it doesn't.
    raw = model.predict([(question, t) for t in texts], batch_size=BATCH_SIZE,
                        activation_fn=torch.nn.Identity(), show_progress_bar=False)
    # Round to 6 decimals, then stable sort: same determinism rule as exact_rank (tiny float noise can't swap order,
    # and exact ties fall back to dense rank)
    # float(s): convert from a numpy number to a plain Python number, so the values can be saved to JSON results files and compared normally.
    # Example: raw = array([ 0.2000003, 5.8124519, 5.8124517 ]) -> scores = [0.2, 5.812452, 5.812452]
    scores = [round(float(s), 6) for s in raw]
    # A sort is stable if items with equal sort values keep the order they had before sorting.
    # head   = [A,   B,   C,   D  ]     # search order: A was best
    # scores = [0.2, 5.8, 3.1, 5.8]     # B and D tie at 5.8
    # len(head) = 4 and range(4) → 0, 1, 2, 3 # the positions of A, B, C, D
    # lambda i: -scores[i] is a short function: "given position i, return its score with a minus sign". 
    # Output: order = [1, 3, 2, 0]      # B, D, C, A
    order = sorted(range(len(head)), key=lambda i: -scores[i])  # sorted() is stable
    reordered = [head[i] for i in order] + tail
    return reordered, [scores[i] for i in order] + [None] * len(tail), n_over_budget(model, question, texts)


def download(model_name):
    """One-time online fetch of the weights. Prints the commit hash to pin in PINNED_REVISIONS."""
    os.environ["HF_HUB_OFFLINE"] = "0"  # only this path goes online; huggingface_hub isn't imported yet, so it applies
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_name)  # downloads into ~/.cache/huggingface/hub; uploads nothing
    print(f"downloaded {model_name}\n  pin this:  \"{model_name}\": \"{os.path.basename(path)}\",")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch weights once (the only online step)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    if args.download:
        download(args.model)
    else:
        ap.print_help()
