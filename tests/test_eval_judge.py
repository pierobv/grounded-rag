from grounded_rag.evaluation.judge import judge_answer, split_claims
from grounded_rag.ingest.models import Chunk


def test_split_claims_separates_cited_and_uncited_sentences():
    text = "Use Argon2id with 19 MiB [1]. Add a unique salt [1][2].\nThis is general advice here."

    claims, uncited = split_claims(text)

    assert [(c.text, c.cited) for c in claims] == [
        ("Use Argon2id with 19 MiB .", [1]),
        ("Add a unique salt .", [1, 2]),
    ]
    assert uncited == 1


def test_short_fragments_are_not_claims():
    claims, uncited = split_claims("1.\nExample:\nUse HttpOnly cookies always [1].")

    assert len(claims) == 1 and uncited == 0


class KeywordJudge:
    """Juiz falso: sustentado se a última palavra da frase aparece em algum trecho."""

    def __init__(self):
        self.calls = []

    def judge(self, claim, excerpts):
        self.calls.append((claim, excerpts))
        word = claim.rstrip(" .").split()[-1].lower()
        return any(word in e.lower() for e in excerpts), "keyword"


def chunk(text):
    return Chunk(id=text, source="s.md", location="L", text=text, strategy="structure")


def test_judge_answer_checks_each_claim_against_cited_excerpts():
    chunks = [chunk("Argon2id needs 19 MiB"), chunk("Salt every password")]
    judge = KeywordJudge()

    verdicts, uncited = judge_answer(
        "Memory is 19 MiB [1]. Paris is lovely in spring [2].", chunks, judge
    )

    assert [v.supported for v in verdicts] == [True, False]
    assert judge.calls[1][1] == ["Salt every password"]  # só o trecho citado vai ao juiz
    assert uncited == 0


def test_citation_to_missing_excerpt_is_not_judged():
    verdicts, _ = judge_answer("Something important here [9].", [chunk("x")], KeywordJudge())

    assert verdicts == []
