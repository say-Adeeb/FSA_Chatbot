import logging
import time
from functools import lru_cache
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://fullstackacademy.in"
USER_AGENT = "FSA-Chatbot-Indexer/1.0 (+https://fullstackacademy.in)"
CRAWL_DELAY_SECONDS = 1.0

# Links to these are downloadable files, not crawlable pages. Following them
# feeds raw binary bytes through the HTML parser and corrupts the index with
# garbage text (e.g. PDF brochures linked from "Download Curriculum" buttons).
NON_PAGE_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".mp4", ".mp3", ".csv",
)


@lru_cache(maxsize=1)
def _robot_parser() -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(urljoin(BASE_URL, "/robots.txt"))
    try:
        # Fetch with our own identifying User-Agent rather than letting
        # RobotFileParser.read() use urllib's default UA -- some WAFs 403 the
        # generic urllib UA, and robotparser treats a 401/403 fetch as
        # "disallow everything", which would wrongly block the whole site.
        response = requests.get(
            urljoin(BASE_URL, "/robots.txt"), headers={"User-Agent": USER_AGENT}, timeout=10
        )
        response.raise_for_status()
        parser.parse(response.text.splitlines())
    except Exception:
        # If robots.txt is unreachable, default to allowing (fail open) rather
        # than blocking ingestion entirely.
        pass
    return parser


def is_allowed(url: str) -> bool:
    try:
        return _robot_parser().can_fetch(USER_AGENT, url)
    except Exception:
        return True


def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        raise ValueError(f"Not an HTML page (Content-Type: {content_type!r}): {url}")

    return response.text


def clean_soup(soup):
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return soup


def extract_text(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    soup = clean_soup(soup)

    title = soup.title.string.strip() if soup.title else "No Title"
    content = soup.get_text(separator=" ", strip=True)
    content = " ".join(content.split())

    return {
        "title": title,
        "content": content
    }


def get_internal_links(html: str, current_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        full_url = urljoin(current_url, href)
        clean_url = full_url.split("#")[0]

        if urlparse(full_url).netloc != urlparse(BASE_URL).netloc:
            continue
        if clean_url.lower().endswith(NON_PAGE_EXTENSIONS):
            continue

        links.add(clean_url)

    return list(links)


def scrape_url(url: str) -> dict:
    if not is_allowed(url):
        raise PermissionError(f"Blocked by robots.txt: {url}")
    html = fetch_page(url)
    data = extract_text(html)
    data["url"] = url
    return data


def scrape_website(start_url: str = BASE_URL, limit: int = 10):
    visited = set()
    queue = [start_url]
    results = []

    while queue and len(results) < limit:
        url = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        if not is_allowed(url):
            logger.info("Skipping (robots.txt disallows): %s", url)
            continue

        try:
            html = fetch_page(url)
            data = extract_text(html)
            data["url"] = url

            results.append(data)

            new_links = get_internal_links(html, url)

            for link in new_links:
                if link not in visited:
                    queue.append(link)

            time.sleep(CRAWL_DELAY_SECONDS)

        except Exception as e:
            logger.warning("Skipped %s: %s", url, e)

    return results