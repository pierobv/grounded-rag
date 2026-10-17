from grounded_rag.ingest.clean import clean_markdown, clean_pdf, normalize


def test_links_keep_text_and_drop_url():
    assert clean_markdown("veja [o guia](https://owasp.org/x)") == "veja o guia"


def test_images_and_comments_are_removed():
    text = "a ![diagrama](img/x.png) b <img src='y.png'> c <!-- nota\ninterna --> d"

    assert clean_markdown(text) == "a b c d"


def test_code_indentation_is_preserved():
    code = "```python\ndef f():\n    return 1\n```"

    assert "    return 1" in clean_markdown(code)


def test_normalize_collapses_spaces_and_blank_lines():
    assert normalize("a   b\n\n\n\nc  ") == "a b\n\nc"


def test_normalize_unifies_pdf_ligatures():
    assert normalize("ﬁle") == "file"  # "ﬁ" é um único caractere (ligadura)


def test_pdf_rejoins_hyphenated_words():
    assert clean_pdf("input vali-\ndation") == "input validation"
