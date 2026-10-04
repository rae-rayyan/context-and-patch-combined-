"""Offline tests for app/services/llm.py (no network, no API key).

Run: pytest test_llm.py -v
xfail(strict=True) = known gap. When you fix it, the test flips to XPASS and
fails the suite, reminding you to delete the xfail marker.
"""
import sys
import types

import pytest

from app.services.llm import OpenAIPatchLLM, _parse_json_object


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '  \n{"a": 1}\n  ',
        '```json\n{"a": 1}\n```',
        '```JSON\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the patch:\n{"a": 1}\nHope that helps!',
        'Here is the patch:\n```json\n{"a": 1}\n```\nDone.',
    ],
)
def test_parses_valid_variants(text):
    assert _parse_json_object(text) == {"a": 1}


def test_parses_nested_object():
    text = '{"edits": [{"path": "a.py", "new": "x = {1: 2}"}]}'
    assert _parse_json_object(text)["edits"][0]["path"] == "a.py"


@pytest.mark.parametrize(
    "text",
    ["", "   ", "no json here", "{not json}", '{"a": 1', "[1, 2]", "null", "42"],
)
def test_rejects_bad_output(text):
    with pytest.raises(ValueError):
        _parse_json_object(text)


@pytest.mark.xfail(strict=True, reason="list-wrapped object is silently unwrapped")
def test_list_wrapped_object_is_rejected():
    with pytest.raises(ValueError):
        _parse_json_object('[{"a": 1}]')


@pytest.mark.xfail(strict=True, reason="brace in prose before the JSON breaks fallback")
def test_braces_in_prose_before_json():
    assert _parse_json_object('Use {braces} like so: {"a": 1}') == {"a": 1}


@pytest.mark.xfail(strict=True, reason="brace in prose after the JSON breaks fallback")
def test_braces_in_prose_after_json():
    assert _parse_json_object('{"a": 1}\nNote: {see above}') == {"a": 1}


# --------------------------------------------------------------- adapter
class _FakeResponses:
    def __init__(self, text):
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return types.SimpleNamespace(output_text=self.text)


def _make_llm(monkeypatch, text):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    fake = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.responses = _FakeResponses(text)

    fake.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    return OpenAIPatchLLM()


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIPatchLLM()


def test_missing_openai_package(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "openai", None)  # makes import raise ImportError
    with pytest.raises(RuntimeError, match="openai"):
        OpenAIPatchLLM()


def test_generate_patch_returns_dict(monkeypatch):
    llm = _make_llm(monkeypatch, '```json\n{"summary": "ok", "edits": []}\n```')
    assert llm.generate_patch("fix it") == {"summary": "ok", "edits": []}


def test_generate_patch_sends_prompt_and_model(monkeypatch):
    llm = _make_llm(monkeypatch, "{}")
    llm.generate_patch("PROMPT-123")
    sent = llm._client.responses.kwargs
    assert sent["model"] == llm.model
    assert any(m["content"] == "PROMPT-123" for m in sent["input"])


def test_empty_model_output_raises_value_error(monkeypatch):
    llm = _make_llm(monkeypatch, "")
    with pytest.raises(ValueError):
        llm.generate_patch("fix it")


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    llm = _make_llm(monkeypatch, "{}")
    assert llm.model == "my-model"


@pytest.mark.xfail(strict=True, reason="api_key is a dataclass field with repr=True")
def test_api_key_not_leaked_in_repr(monkeypatch):
    llm = _make_llm(monkeypatch, "{}")
    assert "sk-test-secret" not in repr(llm)
