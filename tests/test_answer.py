from conftest import FakeLLM

from grounded_rag.generation.answer import Final, Retrieved, TextDelta, stream_answer
from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.base import SearchResult, to_results

CHUNKS = [
    Chunk(
        id="a",
        source="owasp/Session.md",
        location="Cookies",
        text="Set HttpOnly.",
        strategy="structure",
    ),
    Chunk(
        id="b", source="owasp/Session.md", location="IDs", text="Rotate IDs.", strategy="structure"
    ),
]


class FixedRetriever:
    def __init__(self, chunks, score=1.0):
        self.results = to_results([(c, score) for c in chunks])

    def search(self, query: str, k: int) -> list[SearchResult]:
        return self.results[:k]


def run(llm, retriever=None, min_score=None):
    events = list(stream_answer("How?", retriever or FixedRetriever(CHUNKS), llm, 5, min_score))
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    return events, text, events[-1]


def test_event_order_and_grounded_answer():
    events, text, final = run(FakeLLM("Use HttpOnly [1]."))

    assert isinstance(events[0], Retrieved)
    assert isinstance(final, Final)
    assert text == "Use HttpOnly [1]."  # o texto chega em pedaços, mas completo
    assert [c.source for c in final.answer.citations] == ["owasp/Session.md"]
    assert final.stats.prompt_tokens == 100
    assert final.stats.time_to_first_token_ms is not None


def test_prompt_contains_retrieved_excerpts(fake_llm):
    run(fake_llm)

    user_message = fake_llm.calls[0][1]["content"]
    assert "Set HttpOnly." in user_message and "Rotate IDs." in user_message


def test_sentinel_is_not_streamed_to_the_user():
    _, text, final = run(FakeLLM("INSUFFICIENT_CONTEXT The documents do not say.", piece=3))

    assert text == ""  # nada de "INSUFFICIENT_CONTEXT" na tela
    assert final.answer.abstained
    assert final.answer.text == "Não encontrei essa informação nos documentos."


def test_text_that_only_looks_like_sentinel_at_first_is_released():
    _, text, _ = run(FakeLLM("INSecure cookies are bad [1].", piece=2))

    assert text == "INSecure cookies are bad [1]."


def test_uncited_text_is_streamed_but_final_answer_abstains():
    _, text, final = run(FakeLLM("Paris is the capital."))

    assert text == "Paris is the capital."
    assert final.answer.abstained


def test_no_results_abstains_without_calling_llm(fake_llm):
    _, _, final = run(fake_llm, retriever=FixedRetriever([]))

    assert final.answer.abstained
    assert fake_llm.calls == []
    assert final.stats.time_to_first_token_ms is None


def test_low_score_gate_abstains_without_calling_llm(fake_llm):
    _, _, final = run(fake_llm, retriever=FixedRetriever(CHUNKS, score=-3.0), min_score=0.0)

    assert final.answer.abstained
    assert fake_llm.calls == []


def test_score_above_gate_calls_llm(fake_llm):
    run(fake_llm, retriever=FixedRetriever(CHUNKS, score=2.0), min_score=0.0)

    assert len(fake_llm.calls) == 1
