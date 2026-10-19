"""Convert every non-JPG image in journal_images/ to JPG (macOS sips, local), then delete the original.

    python convert_to_jpg.py --dry-run   # list what would happen, change nothing
    python convert_to_jpg.py             # convert; each original is deleted only after its JPG is verified
"""
import argparse
import subprocess
from pathlib import Path

IMAGES_DIR = Path(__file__).parent.parent / "Journal-Entires" / "journal_images"
JPG_EXTS = {".jpg"}  # already the target format: left alone
CONVERT_EXTS = {".png", ".heic", ".heif", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}  # sips can read these; anything else is skipped
JPG_QUALITY = "95"  # high, because OCR reads fine strokes and JPEG is lossy


def dimensions(path):
    """(width, height) in pixels via sips, or None if sips can't read the file."""
    out = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)], capture_output=True, text=True
    )
    if out.returncode != 0:
        return None
    # sips prints "  pixelWidth: 4032" lines; parse "key: value" pairs and ignore the filename line
    vals = {ln.split(":")[0].strip(): ln.split(":")[1].strip() for ln in out.stdout.splitlines() if ":" in ln}
    try:
        return int(vals["pixelWidth"]), int(vals["pixelHeight"])
    except (KeyError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show actions without converting or deleting")
    args = parser.parse_args()

    files = sorted(p for p in IMAGES_DIR.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        # an empty listing would otherwise fail silently
        raise FileNotFoundError(f"No files in {IMAGES_DIR.resolve()}")

    unknown = [p.name for p in files if p.suffix.lower() not in JPG_EXTS | CONVERT_EXTS]
    for name in unknown:
        print(f"SKIP (unknown type, left alone): {name}")

    converted = failed = 0
    for src in (p for p in files if p.suffix.lower() in CONVERT_EXTS):
        dst = src.with_suffix(".jpg")  # same stem, so the filename keeps matching the Notion date_title_N pattern
        if dst.exists():
            # never overwrite: two files for the same photo needs a human decision
            print(f"SKIP (target exists): {src.name}")
            failed += 1
            continue
        if args.dry_run:
            print(f"WOULD CONVERT: {src.name} -> {dst.name}")
            continue

        subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", JPG_QUALITY, str(src), "--out", str(dst)],
            capture_output=True,
        )
        # Verify before deleting: file exists, is non-empty, and has the same pixel size as the original
        # (chained comparison: dims(dst) == dims(src) AND dims(src) is not None, so an unreadable original never passes)
        ok = dst.exists() and dst.stat().st_size > 0 and dimensions(dst) == dimensions(src) is not None
        if not ok:
            print(f"FAILED (original kept): {src.name}")
            dst.unlink(missing_ok=True)  # remove a half-written JPG so a re-run starts clean
            failed += 1
            continue

        src.unlink()
        converted += 1
        print(f"OK: {src.name} -> {dst.name}")

    print(f"\nconverted {converted}, failed/skipped {failed}, unknown types {len(unknown)}")


if __name__ == "__main__":
    main()
