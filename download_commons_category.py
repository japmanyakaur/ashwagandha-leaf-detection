"""Downloads every image in a Wikimedia Commons category, using the public
MediaWiki API (no scraping, no auth needed -- this is the documented way to
enumerate category members: https://www.mediawiki.org/wiki/API:Categorymembers).

Usage:
    python3 download_commons_category.py
"""

import time
from pathlib import Path
from urllib.parse import urlparse

import requests

CATEGORY = "Category:Physalis_peruviana"
OUT_DIR = Path("data/raw/negatives/physalis_peruviana")
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "ashwagandha-leaf-detection-student-project/1.0"}


def get_category_files(category):
    files = []
    cmcontinue = None
    while True:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": category, "cmtype": "file", "cmlimit": "500",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = requests.get(API, params=params, headers=HEADERS).json()
        files.extend(m["title"] for m in data["query"]["categorymembers"])
        if "continue" in data:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            return files


BATCH_SIZE = 50  # MediaWiki's max titles-per-query for anonymous requests --
                  # the whole reason this hit a 429 earlier was doing one API
                  # call per file (167 calls) instead of batching them


def get_file_urls_batch(titles, session, retries=3):
    """Looks up imageinfo for up to BATCH_SIZE titles in a single API call.
    Returns {title: url}, silently omitting any title MediaWiki didn't return
    (e.g. deleted/renamed files)."""
    params = {
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url", "format": "json",
    }
    for attempt in range(1, retries + 1):
        resp = session.get(API, params=params, headers=HEADERS, timeout=15)
        try:
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            print(f"    (attempt {attempt}/{retries}) bad response: "
                  f"status={resp.status_code}, body={resp.text[:200]!r}")
            if attempt == retries:
                return {}
            time.sleep(5 * attempt)  # back off harder -- a 429 means slow down
            continue
        return {
            page["title"]: page["imageinfo"][0]["url"]
            for page in data["query"]["pages"].values()
            if "imageinfo" in page
        }
    return {}


def download_image(url, session, retries=2):
    """Downloads the actual image bytes, validating the response is really an
    image before returning it -- upload.wikimedia.org returns a 429 HTML error
    page (not a normal JSON error) when rate limited, and blindly saving
    whatever comes back is how the last run ended up with 166 HTML files
    named *.jpg instead of real photos."""
    for attempt in range(1, retries + 1):
        resp = session.get(url, headers=HEADERS, timeout=30)
        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 200 and content_type.startswith("image/"):
            return resp.content
        if resp.status_code == 429:
            wait = int(resp.headers.get("retry-after", 60))
            print(f"    rate limited on image download -- waiting {wait}s "
                  f"(Wikimedia's own Retry-After) before continuing")
            time.sleep(wait)
            continue
        print(f"    bad download: status={resp.status_code}, "
              f"content-type={content_type!r}")
        return None
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    titles = get_category_files(CATEGORY)
    print(f"Found {len(titles)} files in {CATEGORY}")

    saved, no_info, wrong_type, already_have, failed = 0, 0, 0, 0, 0
    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i:i + BATCH_SIZE]
        urls = get_file_urls_batch(batch, session)
        for title in batch:
            url = urls.get(title)
            if url is None:
                print(f"  no imageinfo for {title!r}, skipping")
                no_info += 1
                continue
            # url has tracking params appended (...jpg?utm_source=...) -- take
            # the extension from the URL path, not the raw string, or the
            # query string gets mistaken for the extension and every file
            # silently fails this check
            ext = urlparse(url).path.rsplit(".", 1)[-1].lower()
            if ext not in ("jpg", "jpeg", "png"):
                wrong_type += 1
                continue  # skip svg/pdf/etc -- not photos
            fname = OUT_DIR / title.replace("File:", "").replace(" ", "_")
            if fname.exists():
                already_have += 1
                continue
            content = download_image(url, session)
            if content is None:
                print(f"  failed to download {fname.name}, skipping")
                failed += 1
                continue
            fname.write_bytes(content)
            saved += 1
            print(f"  saved {fname.name}")
            time.sleep(1)  # slower pace this time -- 0.3s was still enough to get rate limited
        time.sleep(1)  # pause between metadata batches too

    print(f"\nDone. {saved} photos saved to {OUT_DIR} "
          f"({no_info} had no imageinfo, {wrong_type} weren't jpg/png, "
          f"{already_have} already existed, {failed} failed to download)")


if __name__ == "__main__":
    main()
