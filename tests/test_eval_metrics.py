import pytest

from grounded_rag.evaluation.dataset import RelevantSection
from grounded_rag.evaluation.metrics import (
    bootstrap_ci,
    calibrate_threshold,
    citation_precision,
    first_relevant_rank,
    is_relevant,
    key_fact_recall,
    recall_at_k,
    reciprocal_rank,
)

PW = "owasp/Password_Storage.md"
REL = [RelevantSection(source=PW, section="Password Storage › Argon2id")]
MULTI = [
    RelevantSection(source=PW, section="Password Storage › Argon2id"),
    RelevantSection(source=PW, section="Password Storage › Salting"),
]


def test_subsection_counts_as_relevant():
    assert is_relevant((PW, "Password Storage › Argon2id › Parameters"), REL)


def test_sibling_with_same_prefix_is_not_relevant():
    # "Argon2idX" começa com "Argon2id" como texto, mas é outra seção.
    assert not is_relevant((PW, "Password Storage › Argon2idX"), REL)


def test_other_source_is_not_relevant():
    assert not is_relevant(("owasp/Other.md", "Password Storage › Argon2id"), REL)


def test_fixed_chunk_spanning_sections_is_relevant_if_any_part_matches():
    assert is_relevant((PW, "Password Storage › Intro; Password Storage › Argon2id"), REL)


HITS = [
    (PW, "Password Storage › Intro"),
    (PW, "Password Storage › Argon2id"),
    (PW, "Password Storage › Salting"),
]


def test_first_relevant_rank_and_reciprocal_rank():
    assert first_relevant_rank(HITS, REL) == 2
    assert reciprocal_rank(HITS, REL, k=10) == 0.5
    assert reciprocal_rank(HITS, REL, k=1) == 0.0  # relevante só aparece depois do corte


def test_recall_counts_covered_sections():
    assert recall_at_k(HITS, MULTI, k=1) == 0.0
    assert recall_at_k(HITS, MULTI, k=2) == 0.5  # cobriu Argon2id
    assert recall_at_k(HITS, MULTI, k=3) == 1.0  # cobriu as duas


def test_key_fact_recall_is_case_insensitive():
    assert key_fact_recall("Use 19 MiB and t=2", ["19 mib", "p=1"]) == 0.5
    assert key_fact_recall("anything", []) is None


def test_citation_precision():
    assert citation_precision([HITS[1], HITS[0]], REL) == 0.5
    assert citation_precision([], REL) is None


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [1.0] * 30 + [0.0] * 10  # média 0,75

    low, high = bootstrap_ci(values)

    assert low < 0.75 < high
    assert bootstrap_ci(values) == (low, high)  # mesma semente, mesmo intervalo


def test_bootstrap_ci_of_constant_values_has_no_width():
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0)


def test_calibrate_threshold_separates_scores():
    positives = [5.0, 6.0, 7.0, 2.0]
    negatives = [-3.0, -1.0, 1.0]

    threshold = calibrate_threshold(positives, negatives)

    # Limite 2.0: mantém as 4 respondíveis e barra as 3 fora do corpus.
    assert threshold == pytest.approx(2.0)
