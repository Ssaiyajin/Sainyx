import json

from generation.text.retrieval import (
    QAStore, get_default_store, handwritten_entries, lead_to_sentences, make_answer,
    normalize, parse_qa_pairs, strip_question,
)

GOKU_LEAD = ("Son Goku (Japanese: 孫 悟空, Hepburn: Son Gokū), born Kakarot, is the main protagonist of the "
             "Dragon Ball franchise created by Akira Toriyama. He is a Saiyan raised on Earth. "
             "He later marries Chi-Chi and has two sons.")


def _store():
    s1 = lead_to_sentences(GOKU_LEAD)
    s2 = lead_to_sentences("Dragon Ball is a Japanese media franchise created by Akira Toriyama in 1984. "
                           "It began as a manga series serialized in Weekly Shōnen Jump.")
    s3 = lead_to_sentences("Elden Ring is a 2022 action role-playing game developed by FromSoftware. "
                           "It was directed by Hidetaka Miyazaki with worldbuilding by George R. R. Martin.")
    return QAStore([
        {"title": "Who is Vegeta?", "answer": "Vegeta is the Prince of all Saiyans.", "source": "handwritten"},
        {"title": "Goku", "aliases": ["Kakarot"], "answer": make_answer(s1), "facts": s1, "source": "wikipedia"},
        {"title": "Dragon Ball", "answer": make_answer(s2), "facts": s2, "source": "wikipedia"},
        {"title": "Elden Ring", "answer": make_answer(s3), "facts": s3, "source": "wikipedia"},
    ])


def test_normalize_and_strip_question():
    assert normalize("Gokū's  Power!") == "gokus power"
    assert strip_question(normalize("Who is the Goku?")) == "goku"
    assert strip_question(normalize("What's a Super Saiyan?")) == "super saiyan"
    assert strip_question(normalize("tell me more about Elden Ring.")) == "elden ring"


def test_same_answer_for_every_phrasing():
    store = _store()
    for q in ["goku", "Goku?", "who is goku?", "Who's Goku", "tell me about Goku.", "what is goku", "kakarot", "Gokū"]:
        a = store.answer(q)
        assert a is not None and a.title == "Goku", q
        assert "main protagonist of the Dragon Ball franchise" in a.text


def test_small_typo_and_partial_title():
    store = _store()
    assert store.answer("who is gokuu").title == "Goku"
    assert store.answer("dragon").title == "Dragon Ball"


def test_handwritten_question_matches():
    a = _store().answer("Who is Vegeta?")
    assert a.source == "handwritten" and "Prince" in a.text


def test_passage_question_returns_the_matching_sentence():
    a = _store().answer("who created dragon ball")
    assert a is not None and "Akira Toriyama" in a.text
    a = _store().answer("who directed elden ring")
    assert a is not None and "Miyazaki" in a.text


def test_unknown_questions_are_declined():
    store = _store()
    for q in ["what is the capital of France", "asdf qwer", "how do I bake bread", "", "?"]:
        assert store.answer(q) is None, q


def test_lead_cleanup_and_answer_building():
    sents = lead_to_sentences(GOKU_LEAD)
    assert sents[0].startswith("Son Goku, born Kakarot, is the main protagonist")
    assert "Japanese" not in sents[0]
    # the "R. R." initials in a name must not split the sentence
    s = lead_to_sentences("It was directed by Hidetaka Miyazaki with worldbuilding by George R. R. Martin. It sold well.")
    assert s[0].endswith("George R. R. Martin.")
    assert make_answer([]) is None
    assert make_answer(["x" * 500]) is None


def test_parse_qa_pairs_dedupes():
    text = "Question: Who is A?\nAnswer: A is a.\n\nQuestion: Who is A?\nAnswer: A is a.\n\nQuestion: What is B?\nAnswer: B is b."
    assert parse_qa_pairs(text) == [("Who is A?", "A is a."), ("What is B?", "B is b.")]


def test_repo_handwritten_pairs_answer_goku():
    entries = handwritten_entries()
    assert len(entries) >= 40, "hand-written pairs should be read from generation/text/build_qa.py"
    a = QAStore(entries).answer("who is goku?")
    assert a is not None and "Saiyan" in a.text


def test_default_store_loads_json_when_present(tmp_path):
    p = tmp_path / "store.json"
    p.write_text(json.dumps({"version": 1, "entries": [
        {"title": "Hollow Knight", "answer": "Hollow Knight is a game.", "facts": [], "source": "wikipedia"}]}))
    import generation.text.retrieval as r
    r._default = None
    try:
        store = get_default_store(store_path=str(p))
        assert store.answer("what is hollow knight").title == "Hollow Knight"
        assert store.answer("who is goku").source == "handwritten"
    finally:
        r._default = None