import pymupdf

from grounded_rag.ingest.readers import read_document, read_markdown, read_pdf


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_markdown_location_is_heading_path(tmp_path):
    md = write(
        tmp_path / "doc.md",
        "# XSS\n\nintro\n\n## Encoding\n\nencode\n\n### HTML\n\nhtml\n\n## CSP\n\ncsp\n",
    )

    segments = read_markdown(md, source="doc.md")

    assert [(s.location, s.text) for s in segments] == [
        ("XSS", "intro"),
        ("XSS › Encoding", "encode"),
        ("XSS › Encoding › HTML", "html"),
        ("XSS › CSP", "csp"),  # "## CSP" fecha "## Encoding" e o "### HTML" abaixo dele
    ]


def test_hash_inside_code_block_is_not_a_heading(tmp_path):
    md = write(
        tmp_path / "doc.md",
        "# Setup\n\n```bash\n# instala dependências\npip install x\n```\n",
    )

    segments = read_markdown(md, source="doc.md")

    assert len(segments) == 1
    assert segments[0].location == "Setup"
    assert "# instala dependências" in segments[0].text


def test_text_before_first_heading_uses_file_name(tmp_path):
    md = write(tmp_path / "Notes.md", "sem título\n\n# A\n\ntexto\n")

    segments = read_markdown(md, source="Notes.md")

    assert segments[0].location == "Notes"


def test_empty_sections_are_skipped(tmp_path):
    md = write(tmp_path / "doc.md", "# A\n\n## B\n\nconteúdo\n")

    segments = read_markdown(md, source="doc.md")

    assert [s.location for s in segments] == ["A › B"]


def make_pdf(path, pages):
    """Gera um PDF real no próprio teste: nenhum binário versionado no repositório."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


def test_pdf_has_one_segment_per_page_numbered_from_one(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", ["primeira página", "", "terceira página"])

    segments = read_pdf(pdf, source="doc.pdf")

    # A página 2 está em branco e é ignorada, mas a numeração continua fiel ao PDF.
    assert [s.location for s in segments] == ["p. 1", "p. 3"]
    assert "terceira" in segments[1].text


def test_read_document_uses_path_relative_to_root(tmp_path):
    (tmp_path / "owasp").mkdir()
    md = write(tmp_path / "owasp" / "XSS.md", "# XSS\n\ntexto\n")

    segments = read_document(md, root=tmp_path)

    assert segments[0].source == "owasp/XSS.md"
