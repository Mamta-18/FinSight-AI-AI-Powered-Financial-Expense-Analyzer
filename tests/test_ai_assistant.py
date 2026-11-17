"""Tests for provider selection, financial context, and safe fallback handling."""

import pandas as pd
import sys
import types

from modules import ai_assistant


def test_groq_model_default_and_environment_override(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert ai_assistant._get_groq_model() == "openai/gpt-oss-20b"

    monkeypatch.setenv("GROQ_MODEL", " custom-model-id ")
    assert ai_assistant._get_groq_model() == "custom-model-id"


def _configured_groq(monkeypatch):
    settings = {
        "llm_provider": "groq",
        "groq_api_key": "test-groq-key",
        "openai_api_key": "test-openai-key",
    }
    monkeypatch.setattr(ai_assistant, "get_setting", lambda key, default=None: settings.get(key, default))
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)


def test_selected_groq_provider_is_used_and_financial_summary_is_sent(monkeypatch):
    _configured_groq(monkeypatch)
    captured = {}

    def fake_groq(messages, api_key):
        captured["messages"] = messages
        captured["api_key"] = api_key
        return "Your summarized answer."

    monkeypatch.setattr(ai_assistant, "_call_groq", fake_groq)
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-05", "2025-01-10"]),
            "description": ["private vendor detail", "private salary detail"],
            "amount": [250.0, 1000.0],
            "type": ["expense", "income"],
            "category": ["Food & Dining", "Income"],
        }
    )

    answer, provider = ai_assistant.ask_assistant(
        "What is my spending?", transactions, []
    )

    context = captured["messages"][0]["content"]
    assert answer == "Your summarized answer."
    assert provider == "groq"
    assert captured["api_key"] == "test-groq-key"
    assert "Total Expenses: ₹250" in context
    assert "Food & Dining: ₹250" in context
    assert "private vendor detail" not in context
    assert "private salary detail" not in context


def test_missing_groq_sdk_error_is_actionable_without_exposing_key(monkeypatch):
    _configured_groq(monkeypatch)

    def missing_sdk(messages, api_key):
        raise ModuleNotFoundError("No module named 'groq'", name="groq")

    monkeypatch.setattr(ai_assistant, "_call_groq", missing_sdk)
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-05"]),
            "description": ["private transaction"],
            "amount": [250.0],
            "type": ["expense"],
            "category": ["Food & Dining"],
        }
    )

    answer, provider = ai_assistant.ask_assistant("What did I spend?", transactions, [])

    assert provider == "analytical (fallback)"
    assert "Groq SDK is missing" in answer
    assert "requirements.txt" in answer
    assert "test-groq-key" not in answer


def test_selected_provider_does_not_silently_switch_to_another_key(monkeypatch):
    settings = {"llm_provider": "groq", "openai_api_key": "test-openai-key"}
    monkeypatch.setattr(ai_assistant, "get_setting", lambda key, default=None: settings.get(key, default))
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)

    assert ai_assistant._get_llm_provider() is None


def test_connection_check_makes_selected_provider_request(monkeypatch):
    _configured_groq(monkeypatch)
    calls = []
    monkeypatch.setattr(
        ai_assistant, "_call_groq", lambda messages, api_key: calls.append((messages, api_key)) or "OK"
    )

    success, message = ai_assistant.test_llm_connection()

    assert success is True
    assert "Groq API request succeeded" in message
    assert len(calls) == 1
    assert calls[0][0][-1] == {"role": "user", "content": "Connection test. Reply OK."}


def test_groq_chat_request_uses_configured_model(monkeypatch):
    calls = {}

    class Completions:
        def create(self, **kwargs):
            calls.update(kwargs)
            message = types.SimpleNamespace(content="OK")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions())
    )
    fake_groq_module = types.SimpleNamespace(Groq=lambda api_key: fake_client)
    monkeypatch.setitem(sys.modules, "groq", fake_groq_module)
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    result = ai_assistant._call_groq([{"role": "user", "content": "test"}], "not-logged")

    assert result == "OK"
    assert calls["model"] == "llama-3.3-70b-versatile"


def test_groq_chat_retries_next_model_after_model_not_found(monkeypatch):
    attempted_models = []

    class NotFound(Exception):
        status_code = 404

    class Completions:
        def create(self, **kwargs):
            attempted_models.append(kwargs["model"])
            if len(attempted_models) == 1:
                raise NotFound()
            message = types.SimpleNamespace(content="Available model responded.")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions())
    )
    monkeypatch.setitem(
        sys.modules, "groq", types.SimpleNamespace(Groq=lambda api_key: fake_client)
    )
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    result = ai_assistant._call_groq([{"role": "user", "content": "test"}], "not-logged")

    assert result == "Available model responded."
    assert attempted_models == [
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]


def test_api_key_resolution_trims_and_prefers_environment(monkeypatch):
    settings = {"groq_api_key": "  sqlite-key  "}
    monkeypatch.setattr(ai_assistant, "get_setting", lambda key, default=None: settings.get(key, default))
    monkeypatch.setenv("GROQ_API_KEY", "  env-key\n")

    key, source = ai_assistant._resolve_api_key("groq")

    assert key == "env-key"
    assert source == "environment variable"


def test_connection_401_identifies_key_source_without_exposing_key(monkeypatch):
    _configured_groq(monkeypatch)

    class Unauthorized(Exception):
        status_code = 401

    monkeypatch.setattr(
        ai_assistant, "_call_groq", lambda messages, api_key: (_ for _ in ()).throw(Unauthorized())
    )

    success, message = ai_assistant.test_llm_connection()

    assert success is False
    assert "HTTP 401" in message
    assert "from Settings" in message
    assert "Replace it" in message
    assert "test-groq-key" not in message


def test_network_failure_has_distinct_user_message():
    class APIConnectionError(Exception):
        pass

    message = ai_assistant._safe_provider_error("groq", APIConnectionError())

    assert "network or timeout" in message


def test_unauthorized_provider_keeps_analytical_fallback(monkeypatch):
    _configured_groq(monkeypatch)

    class Unauthorized(Exception):
        status_code = 401

    monkeypatch.setattr(
        ai_assistant, "_call_groq", lambda messages, api_key: (_ for _ in ()).throw(Unauthorized())
    )
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-05", "2025-01-10"]),
            "description": ["private vendor detail", "private salary detail"],
            "amount": [250.0, 1000.0],
            "type": ["expense", "income"],
            "category": ["Food & Dining", "Income"],
        }
    )

    answer, provider = ai_assistant.ask_assistant(
        "Which category is the highest expense?", transactions, []
    )

    assert provider == "analytical (fallback)"
    assert "HTTP 401" in answer
    assert "Food & Dining" in answer
    assert "₹250" in answer
    assert "private vendor detail" not in answer
