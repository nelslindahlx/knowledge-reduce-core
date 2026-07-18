"""
Tests for multi-format document ingestion.

ingest.load_text() turns a file into plain text regardless of format:
.txt passes through, .md strips light markup, .html extracts the body
text (including Substack-style body_html in embedded JSON), and .pdf is
supported when the optional pymupdf extra is installed.
"""
import pytest

from knowledge_graph_pkg.ingest import load_text, html_to_text, markdown_to_text


# ---------- plain text ----------

def test_txt_passthrough(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Marie Curie discovered radium.")
    assert "Marie Curie discovered radium." in load_text(str(p))


# ---------- markdown ----------

def test_markdown_strips_markup():
    md = "# Heading\n\n- **Robert Putnam** wrote *Bowling Alone*.\n\n[link](http://x.com)"
    out = markdown_to_text(md)
    assert "Robert Putnam wrote Bowling Alone." in out
    assert "#" not in out
    assert "*" not in out
    assert "](" not in out  # link syntax gone


def test_md_file_via_load_text(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("## Title\n\nMarie Curie discovered **radium**.")
    out = load_text(str(p))
    assert "Marie Curie discovered radium." in out
    assert "#" not in out


# ---------- html ----------

def test_html_to_text_basic():
    html = "<html><body><h1>Title</h1><p>Robert Putnam wrote Bowling Alone.</p></body></html>"
    out = html_to_text(html)
    assert "Robert Putnam wrote Bowling Alone." in out
    assert "<" not in out


def test_html_paragraph_separation():
    html = "<p>First sentence.</p><p>Second sentence.</p>"
    out = html_to_text(html)
    # paragraphs should not be glued together
    assert "First sentence." in out and "Second sentence." in out
    assert "First sentence.Second sentence." not in out


def test_html_file_via_load_text(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><body><p>Marie Curie discovered radium.</p></body></html>")
    out = load_text(str(p))
    assert "Marie Curie discovered radium." in out


def test_html_substack_body_html_json():
    # Substack embeds the post body as escaped JSON in window._preloads.
    body = "<p>Nels Lindahl wrote Civic Honors.</p>"
    page = (
        '<html><head><script>window._preloads = JSON.parse('
        '"{\\"post\\":{\\"body_html\\":\\"<p>Nels Lindahl wrote Civic Honors.</p>\\"}}"'
        ');</script></head><body>boilerplate nav</body></html>'
    )
    out = html_to_text(page)
    assert "Nels Lindahl wrote Civic Honors." in out


# ---------- errors ----------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text(str(tmp_path / "nope.txt"))
