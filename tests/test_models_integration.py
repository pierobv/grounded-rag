"""Testes com modelos locais reais (cross-encoder baixado do Hugging Face).

Fora do CI; rode localmente com: uv run pytest -m models
"""

import pytest

from grounded_rag.config import Settings
from grounded_rag.retrieval.rerank import CrossEncoderReranker

pytestmark = pytest.mark.models


def test_cross_encoder_prefers_relevant_text():
    reranker = CrossEncoderReranker(Settings().rerank_model)

    relevant, unrelated = reranker.score(
        "How do I prevent clickjacking?",
        [
            "Send the X-Frame-Options header or the CSP frame-ancestors directive so other "
            "sites cannot embed your pages in frames.",
            "Store passwords using Argon2id with a unique salt per user.",
        ],
    )

    assert relevant > unrelated
