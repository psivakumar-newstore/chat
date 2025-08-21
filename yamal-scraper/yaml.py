import os, time, urllib.parse, re
import requests
from bs4 import BeautifulSoup

START_URL = "https://docs.newstore.net/api/"   # root page with links to sections
OUTPUT_DIR = "openapi-yaml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "yaml-fetcher/1.0"})

visited = set()
queue = [START_URL]
domain = urllib.parse.urlparse(START_URL).netloc

def is_yaml_link(href: str) -> bool:
    if not href:
        return False
    href_lower = href.lower()
    return (
        href_lower.endswith(".yaml")
        or href_lower.endswith(".yml")
        or "format=yaml" in href_lower
        or "openapi" in href_lower and ("download" in href_lower or "spec" in href_lower)
    )

def same_domain(url: str) -> bool:
    try:
        return urllib.parse.urlparse(url).netloc == domain
    except Exception:
        return False

def save_yaml(url: str):
    r = session.get(url, timeout=30, allow_redirects=True)
    r.raise_for_status()
    # filename: prefer Content-Disposition; else from URL path; else slug
    fname = None
    cd = r.headers.get("content-disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd, flags=re.I)
    if m:
        fname = m.group(1)
    if not fname:
        path = urllib.parse.urlparse(r.url).path
        fname = os.path.basename(path) or "openapi.yaml"
    if not fname.lower().endswith((".yaml", ".yml")):
        fname += ".yaml"
    out_path = os.path.join(OUTPUT_DIR, fname)
    with open(out_path, "wb") as f:
        f.write(r.content)
    print("Saved:", out_path)

while queue:
    url = queue.pop(0)
    if url in visited:
        continue
    visited.add(url)

    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print("Skip (error):", url, e)
        continue

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1) download any YAML links on this page
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(resp.url, a["href"])
        if is_yaml_link(href):
            try:
                save_yaml(href)
                time.sleep(0.2)  # be polite
            except Exception as e:
                print("Failed:", href, e)

    # 2) crawl deeper within the same domain (docs sections)
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(resp.url, a["href"])
        if same_domain(href) and href not in visited and "#" not in href and any(k in href.lower() for k in ["docs", "api", "explorer"]):
            queue.append(href)