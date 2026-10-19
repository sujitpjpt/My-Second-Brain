import ollama
import time
from pathlib import Path

MODELS = ["qwen2.5vl:7b", "minicpm-v:8b"]
IMAGES_DIR = Path(__file__).parent.parent / "Journal-Entires" / "test_journal_images"
OUTPUT_DIR = Path(__file__).parent.parent / "Journal-Entires" / "ocr_test_output"

PROMPT = (
    "Transcribe all handwritten text in this image exactly as written, "
    "preserving line breaks. Output only the transcription, no commentary."
)


NUM_CTX = 8192
NUM_PREDICT = 2048
REPEAT_PENALTY = 1.15  # >1 discourages loops; Ollama default is ~1.1, keep it small (strong values corrupt real repeats)
REPEAT_LAST_N = 256    # tokens of history the penalty looks at


def check_output(response, text):
    """Return problem flags for one OCR call; empty list means clean."""
    flags = []
    done_reason = response.get("done_reason")
    prompt_toks = response.get("prompt_eval_count") or 0
    resp_toks = response.get("eval_count") or 0

    if not text:
        flags.append("EMPTY(thinking-only)" if response.get("thinking") else "EMPTY")

    if done_reason == "length":
        # which cap was hit: response cap vs total context window
        if resp_toks >= NUM_PREDICT:
            cause = "num_predict"
        elif prompt_toks + resp_toks >= NUM_CTX:
            cause = "num_ctx"
        else:
            cause = "unknown"
        flags.append(f"TRUNCATED({cause})")
    elif prompt_toks + resp_toks > 0.9 * NUM_CTX:
        flags.append("CTX-NEAR-FULL")  # finished, but little headroom left

    # looping model: many lines, few distinct
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.5:
        flags.append("REPETITIVE")

    return flags


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg files found in {IMAGES_DIR.resolve()}")

    for model in MODELS:
        model_dir = OUTPUT_DIR / model.replace(":", "_")
        model_dir.mkdir(exist_ok=True)
        print(f"\n=== {model} ===")
        results = []  # (elapsed, flags) per image

        for image_path in images:
            print(f"processing... {image_path.name}")
            start = time.monotonic()
            response = ollama.generate(
                model=model,
                prompt=PROMPT,
                images=[str(image_path)],
                options={
                    "num_ctx": NUM_CTX,
                    "num_predict": NUM_PREDICT,
                    "repeat_penalty": REPEAT_PENALTY,
                    "repeat_last_n": REPEAT_LAST_N,
                },
                # thinking models spend num_predict on reasoning and return empty text; others reject the flag
                **({"think": False} if model.startswith("qwen3") else {}),
            )
            elapsed = time.monotonic() - start

            text = response["response"].strip()
            out_path = model_dir / f"{image_path.stem}.txt"
            out_path.write_text(text)

            done_reason = response.get("done_reason")
            eval_count = response.get("eval_count")
            prompt_eval_count = response.get("prompt_eval_count")
            flags = check_output(response, text)
            flag = " " + " ".join(f"[{f}]" for f in flags) if flags else " [OK]"
            results.append((elapsed, flags))

            print(
                f"{image_path.name}: {elapsed:.1f}s -> {out_path.name} "
                f"(done_reason={done_reason}, prompt_tokens={prompt_eval_count}, "
                f"response_tokens={eval_count}, chars={len(text)}){flag}"
            )

        clean = sum(1 for _, f in results if not f)
        avg = sum(t for t, _ in results) / len(results)
        print(f"--- {model}: {clean}/{len(results)} clean, avg {avg:.1f}s/image")


if __name__ == "__main__":
    main()