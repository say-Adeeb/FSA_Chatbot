"""Unit tests for pure-logic pieces of the web scraper (no real network calls)."""
from app.ingestion.web_scraper import extract_text, get_internal_links


class TestExtractText:
    def test_strips_script_and_style(self):
        html = "<html><head><title>T</title></head><body><script>bad()</script>hello</body></html>"
        data = extract_text(html)
        assert data["title"] == "T"
        assert "bad()" not in data["content"]
        assert "hello" in data["content"]

    def test_missing_title_defaults(self):
        data = extract_text("<html><body>hi</body></html>")
        assert data["title"] == "No Title"


class TestInternalLinks:
    def test_keeps_only_same_domain_links(self):
        html = (
            '<a href="/about/">About</a>'
            '<a href="https://external.example.com/page">External</a>'
        )
        links = get_internal_links(html, "https://fullstackacademy.in/")
        assert any("fullstackacademy.in/about" in link for link in links)
        assert all("external.example.com" not in link for link in links)

    def test_strips_fragment(self):
        html = '<a href="/about/#team">About Team</a>'
        links = get_internal_links(html, "https://fullstackacademy.in/")
        assert all("#" not in link for link in links)

    def test_excludes_downloadable_file_links(self):
        # Regression: "Download Curriculum" PDF links were being crawled as
        # if they were HTML pages, corrupting the index with raw PDF bytes.
        html = (
            '<a href="/about/">About</a>'
            '<a href="/wp-content/uploads/2025/Course-content.pdf">Download Curriculum</a>'
            '<a href="/images/logo.png">Logo</a>'
        )
        links = get_internal_links(html, "https://fullstackacademy.in/")
        assert any("about" in link for link in links)
        assert not any(link.lower().endswith((".pdf", ".png")) for link in links)
