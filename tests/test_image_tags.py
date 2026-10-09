from generation.image.tags import (
    build_vocab, encode_indices, parse_prompt, resolve_prompt, tag_phrase,
)

VOCAB = [
    "solo", "1boy", "dragon_ball", "son_goku", "vegeta", "gogeta", "vegito",
    "blue_hair", "blonde_hair", "super_saiyan", "super_saiyan_blue",
    "trunks_(dragon_ball)", "red_hair",
]


def test_tag_phrase_strips_parenthetical_and_underscores():
    assert tag_phrase("trunks_(dragon_ball)") == "trunks"
    assert tag_phrase("blue_hair") == "blue hair"


def test_gogeta_blue_maps_to_character_and_form():
    tags, unmatched = parse_prompt("gogeta blue", VOCAB)
    assert tags[0] == "gogeta"
    assert "super_saiyan_blue" in tags and "blue_hair" in tags
    assert unmatched == []


def test_longest_phrase_wins():
    tags, _ = parse_prompt("goku super saiyan blue", VOCAB)
    assert "super_saiyan_blue" in tags
    assert "super_saiyan" not in tags


def test_alias_and_stopwords():
    tags, unmatched = parse_prompt("Generate a picture of Goku", VOCAB)
    assert tags == ["son_goku"]
    assert unmatched == []


def test_unknown_words_are_reported_not_ignored():
    tags, unmatched = parse_prompt("vegeta riding a banana", VOCAB)
    assert tags == ["vegeta"]
    assert "banana" in unmatched and "riding" in unmatched


def test_aliases_only_use_tags_in_vocab():
    tags, _ = parse_prompt("ssb", ["son_goku"])
    assert tags == []


def test_build_vocab_drops_meta_and_rare_tags_and_keeps_characters():
    lists = [["a", "highres", "gogeta"]] * 12 + [["a", "rare"]]
    vocab = build_vocab(lists, character_tags=["gogeta", "missing"], min_count=5)
    assert "highres" not in vocab and "rare" not in vocab
    assert "a" in vocab and "gogeta" in vocab and "missing" not in vocab


def test_encode_indices_ignores_unknown_tags():
    assert encode_indices(["vegeta", "nope"], VOCAB) == [VOCAB.index("vegeta")]


def test_resolve_prompt_unconditional_model_has_no_error():
    class Old:
        pass

    result = resolve_prompt(Old(), "anything at all")
    assert result["conditioned"] is False and result["error"] is None


def test_resolve_prompt_refuses_unknown_prompt_with_suggestions():
    class Model:
        tag_vocab = VOCAB
        character_tags = ["son_goku", "vegeta"]

    result = resolve_prompt(Model(), "a cat")
    assert result["error"] and "vegeta" in result["error"]
    assert result["tags"] == []