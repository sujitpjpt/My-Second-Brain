import argparse
import os
import re
import time
from datetime import datetime

import requests

NOTION_TOKEN = os.environ["NOTION_API_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Journal-Entires", "journal_images")

REJECTED_DIR = os.path.join(SCRIPT_DIR, "..", "Journal-Entires", "rejected_journal_images")

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def iso_date(value):
    """argparse type: strict YYYY-MM-DD that is also a real calendar date."""
    # strptime alone accepts "2026-9-1"; the regex forces zero-padding, so text comparison later is safe
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError(f"'{value}' is not YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")  # rejects e.g. 2026-02-30
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a real calendar date")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Download journal images for Notion pages in a Date range.")
    # Bounds apply to the Notion "Date" field (not created/last-edited) and are inclusive
    parser.add_argument("--start", type=iso_date, required=True, help="earliest page date, YYYY-MM-DD")
    parser.add_argument("--end", type=iso_date, help="latest page date, YYYY-MM-DD (default: no upper bound)")
    args = parser.parse_args()
    if args.end and args.end < args.start:
        parser.error(f"--end {args.end} is before --start {args.start}")
    return args


def in_range(date_str, start, end):
    """True if the page date is inside [start, end]. YYYY-MM-DD strings compare correctly as text."""
    if date_str == "unknown-date":
        return False  # a page with no Date can't be placed in a range
    return date_str >= start and (end is None or date_str <= end)


def slugify(text, max_len=50):
    text = re.sub(r"[^\w\s-]", "", text or "")
    text = re.sub(r"[\s_]+", "-", text.strip())
    return text[:max_len]


def get_all_pages():
    pages = []
    url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
    payload = {"page_size": 100}
    while True:
        resp = requests.post(url, headers=HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        pages.extend(data["results"])
        print(f"  Fetched {len(pages)} pages so far...")
        if not data.get("has_more"):
            break
        payload["start_cursor"] = data["next_cursor"]
    return pages


def get_page_image_urls(page_id):
    urls = []
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    params = {"page_size": 100}
    while True:
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        for block in data["results"]:
            if block["type"] == "image":
                img = block["image"]
                if img["type"] == "file":
                    urls.append(img["file"]["url"])
                elif img["type"] == "external":
                    urls.append(img["external"]["url"])
        if not data.get("has_more"):
            break
        params["start_cursor"] = data["next_cursor"]
    return urls


def extract_page_meta(page):
    props = page.get("properties", {})

    # Date property
    date_str = "unknown-date"
    for key in props:
        p = props[key]
        if p["type"] == "date" and p.get("date"):
            date_str = p["date"]["start"][:10]
            break

    # Name/title property
    title_str = "Untitled"
    for key in props:
        p = props[key]
        if p["type"] == "title":
            parts = p.get("title", [])
            if parts:
                title_str = "".join(t.get("plain_text", "") for t in parts)
            break

    return date_str, title_str


def download_image(url, filepath):
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    ext = None
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "png" in content_type:
        ext = ".png"
    elif "gif" in content_type:
        ext = ".gif"
    elif "webp" in content_type:
        ext = ".webp"
    elif "heic" in content_type:
        ext = ".heic"

    if ext and not filepath.lower().endswith(ext):
        filepath = filepath + ext
    elif not ext:
        # guess from URL
        for candidate in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"]:
            if candidate in url.lower():
                if not filepath.lower().endswith(candidate):
                    filepath = filepath + candidate
                break
        else:
            filepath = filepath + ".jpg"

    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return filepath


def rejected_stems():
    """Filenames (no extension) of images I hand-rejected. Extension-free so .heic/.png -> .jpg conversion still matches."""
    if not os.path.isdir(REJECTED_DIR):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(REJECTED_DIR)}


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rejected = rejected_stems()

    print("Fetching all pages from database...")
    pages = get_all_pages()
    print(f"Total pages found: {len(pages)}\n")

    total_images = 0
    pages_with_images = 0

    for i, page in enumerate(pages, 1):
        page_id = page["id"]
        date_str, title_str = extract_page_meta(page)
        if not in_range(date_str, args.start, args.end):
            continue  # outside the requested range: skip before any block/image fetch
        slug = slugify(title_str)
        prefix = f"{date_str}_{slug}" if slug else date_str

        print(f"[{i}/{len(pages)}] {prefix} — fetching blocks...", end="", flush=True)

        try:
            image_urls = get_page_image_urls(page_id)
        except Exception as e:
            print(f" ERROR fetching blocks: {e}")
            continue

        if not image_urls:
            print(" (no images)")
            continue

        print(f" {len(image_urls)} image(s) — downloading...")
        pages_with_images += 1

        for j, img_url in enumerate(image_urls, 1):
            base_path = os.path.join(OUTPUT_DIR, f"{prefix}_{j}")
            if f"{prefix}_{j}" in rejected:
                print(f"    [{j}] Previously rejected, skipping.")
                continue
            # Skip if already downloaded (any extension)
            existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"{prefix}_{j}")]
            if existing:
                print(f"    [{j}] Already exists, skipping.")
                total_images += 1
                continue
            try:
                saved_path = download_image(img_url, base_path)
                print(f"    [{j}] Saved: {os.path.basename(saved_path)}")
                total_images += 1
            except Exception as e:
                print(f"    [{j}] FAILED: {e}")

        # Small delay to be polite to the API
        time.sleep(0.2)

    print(f"\n{'='*60}")
    print(f"Done! Pages with images: {pages_with_images}")
    print(f"Total images downloaded: {total_images}")
    print(f"Saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
