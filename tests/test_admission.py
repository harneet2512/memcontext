from __future__ import annotations

from memcontext.admission import admit


def test_admit_normal_text():
    result = admit("I prefer dark mode for my editor")
    assert result.admitted is True


def test_reject_empty():
    result = admit("")
    assert result.admitted is False


def test_reject_filler_only():
    result = admit("uh um ok yeah")
    assert result.admitted is False


def test_reject_silence_marker():
    result = admit("[silence]")
    assert result.admitted is False
    result2 = admit("[noise]")
    assert result2.admitted is False
    result3 = admit("[pause]")
    assert result3.admitted is False


def test_reject_too_few_content_words():
    result = admit("ok yeah")
    assert result.admitted is False


def test_admit_system_turn_with_content():
    result = admit("The patient has a history of hypertension and diabetes")
    assert result.admitted is True


def test_admit_short_but_meaningful():
    result = admit("I live in Toronto")
    assert result.admitted is True
