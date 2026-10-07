"""
Extracts href values from every <a class="_3+kae VKeLo tdyTc"> on the
PitVis-2023 UCL RDR article page.

The page renders via JS and sits behind bot-detection that has blocked every
automated fetch tried so far (requests/curl/headless Playwright, from this
particular environment) with a 403, while a normal browser session works
fine. This script will try a live fetch first; if that 403s, it tells you
so and falls back to reading a saved HTML file instead (DevTools -> Elements
panel -> right-click <html> -> Copy -> Copy outerHTML, saved to a file).

Usage:
    python extract_pitvis_links.py <url>          # live fetch attempt
    python extract_pitvis_links.py page.html       # parse a saved file
    cat page.html | python extract_pitvis_links.py -
"""
import sys
import requests
from bs4 import BeautifulSoup

TARGET_CLASSES = {"_3+kae", "VKeLo", "tdyTc"}

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    is_waf_challenge = resp.headers.get("x-amzn-waf-action") == "challenge" or "challenge.js" in resp.text
    if resp.status_code != 200 or is_waf_challenge:
        reason = "WAF JS challenge (needs a real browser)" if is_waf_challenge else f"HTTP {resp.status_code}"
        raise RuntimeError(
            f"Fetch failed: {reason}. This host has been blocking automated requests from this "
            f"environment specifically (403 in some attempts, the WAF challenge in others) -- if "
            f"you're seeing this too, save the page's rendered DOM from your own browser instead "
            f"(see module docstring) and pass that file's path here instead of the URL."
        )
    return resp.text

def extract_hrefs(html: str):
    soup = BeautifulSoup(html, "html.parser")
    hrefs = []
    for a in soup.find_all("a", href=True):
        classes = set(a.get("class", []))
        if TARGET_CLASSES.issubset(classes):
            hrefs.append(a["href"])
    return hrefs

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "https://rdr.ucl.ac.uk/articles/dataset/PitVis_Challenge_Endoscopic_Pituitary_Surgery_videos/26531686"

    if arg == "-":
        html = sys.stdin.read()
    elif arg.startswith("http://") or arg.startswith("https://"):
        html = fetch_html(arg)
    else:
        html = open(arg, "r", encoding="utf-8").read()

    hrefs = extract_hrefs(html)
    print(f"Found {len(hrefs)} matching <a> tag(s):", file=sys.stderr)
    for href in hrefs:
        print(href)
