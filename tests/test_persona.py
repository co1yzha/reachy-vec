import pytest

from reachy_vec.brain.persona import STYLES, Persona


def test_plain_keeps_the_original_lines():
    p = Persona("plain")
    assert p.style_prompt == ""
    assert p.line("greet", name="Alice") == "Hi Alice! What can I help you with?"
    assert p.line("apology") == "Sorry, my brain isn't responding right now."


@pytest.mark.parametrize("style", ["british", "scouse"])
def test_every_style_has_every_line(style):
    plain = STYLES["plain"]["lines"]
    lines = STYLES[style]["lines"]
    assert set(lines) == set(plain)
    for key, variants in lines.items():
        assert variants, key
        for v in variants:
            v.format(name="Alice")  # placeholders are only {name}


def test_greet_uses_the_name_in_every_variant():
    for style in STYLES:
        for v in STYLES[style]["lines"]["greet"]:
            assert "{name}" in v


def test_scouse_prompt_is_light_and_neutral_on_football():
    prompt = Persona("scouse").style_prompt
    assert "Liverpool" in prompt
    assert "Everton" in prompt          # stays neutral, says so
    assert "straight answer" in prompt  # humour never replaces the answer
    assert len(prompt) < 2000           # re-read every turn


def test_unknown_style_falls_back_to_plain():
    assert Persona("klingon").style == "plain"


def test_fillers_come_from_the_style():
    assert Persona("scouse").fillers() == list(STYLES["scouse"]["lines"]["filler"])
