"""
modules/ai_assistant.py
LLM-powered AI Financial Assistant for FinSight AI.
Supports OpenAI, Google Gemini, and Groq.
Raw transactions are NEVER sent to external APIs — only aggregated summaries.
"""

import os
import json
from typing import Optional
import pandas as pd
from modules.financial_analytics import (
    compute_kpis, get_category_expense,
    get_monthly_income_expense, generate_budget_suggestions,
)
from database.db_operations import get_setting


DISCLAIMER = (
    "⚠️ *FinSight AI provides budgeting insights for informational purposes only. "
    "This is not certified financial advice. Always consult a qualified financial advisor for major decisions.*"
)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MODEL_FALLBACKS = (
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)


def _get_groq_model() -> str:
    """Use a configurable Groq model ID, falling back to the supported default."""
    return (os.getenv("GROQ_MODEL") or "").strip() or DEFAULT_GROQ_MODEL


def _get_groq_model_candidates() -> tuple[str, ...]:
    """Return the configured model first, followed by known SDK model IDs."""
    configured = _get_groq_model()
    return tuple(dict.fromkeys((configured, *GROQ_MODEL_FALLBACKS)))


def _build_financial_context(df: pd.DataFrame) -> str:
    """
    Build a compact financial summary (aggregated data only) to use as LLM context.
    Raw transaction descriptions are intentionally omitted for privacy.
    """
    if df.empty:
        return "No financial data is currently available."

    kpis    = compute_kpis(df)
    cat_exp = get_category_expense(df, top_n=8)
    monthly = get_monthly_income_expense(df)

    lines = [
        "=== USER FINANCIAL SUMMARY (Aggregated) ===",
        f"Period: {df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}",
        f"Total Income:   ₹{kpis['total_income']:,.0f}",
        f"Total Expenses: ₹{kpis['total_expense']:,.0f}",
        f"Total Savings:  ₹{kpis['total_savings']:,.0f}",
        f"Savings Rate:   {kpis['savings_rate']}%",
        f"Avg Monthly Expense: ₹{kpis['avg_monthly_expense']:,.0f}",
        "",
        "--- Expense by Category ---",
    ]

    for _, row in cat_exp.iterrows():
        lines.append(f"  {row['category']}: ₹{row['total']:,.0f} ({row['pct']}%)")

    lines.append("")
    lines.append("--- Recent Monthly Expenses (last 6) ---")
    for _, row in monthly.tail(6).iterrows():
        lines.append(
            f"  {row['month_str']}: Income ₹{row['income']:,.0f} | Expense ₹{row['expense']:,.0f} | Savings ₹{row['savings']:,.0f}"
        )

    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are FinSight AI, a friendly and professional AI Financial Assistant.
You help users understand their personal spending patterns and suggest practical ways to save money.

IMPORTANT RULES:
1. Base ALL numerical answers on the financial context provided below.
2. Do NOT invent or estimate numbers not in the context.
3. Keep responses concise, friendly, and actionable (2-4 sentences or a short list).
4. Use ₹ symbol for Indian Rupees.
5. If you cannot answer from the data, say so clearly.
6. Never claim to provide certified financial advice.

{context}
"""


def _get_llm_provider() -> Optional[str]:
    """Return the selected provider when its key is configured; otherwise autodetect only if none is selected."""
    provider = (get_setting("llm_provider", "") or os.getenv("LLM_PROVIDER", "")).strip().lower()
    key_map = {
        "openai": ("OPENAI_API_KEY", "openai_api_key"),
        "google": ("GOOGLE_API_KEY", "google_api_key"),
        "groq": ("GROQ_API_KEY", "groq_api_key"),
    }

    # A configured choice is authoritative. Do not silently use a different
    # provider just because it has a key available.
    if provider in key_map:
        return provider if _get_api_key(provider) else None

    # Auto-detect from available keys
    if _get_api_key("groq"):
        return "groq"
    if _get_api_key("openai"):
        return "openai"
    if _get_api_key("google"):
        return "google"

    return None


def _resolve_api_key(provider: str) -> tuple[Optional[str], Optional[str]]:
    """Return a trimmed configured key and its source without logging its value."""
    env_map = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}
    db_map  = {"openai": "openai_api_key",  "google": "google_api_key",  "groq": "groq_api_key"}
    env_name = env_map.get(provider)
    db_name = db_map.get(provider)
    if not env_name or not db_name:
        return None, None

    env_key = (os.getenv(env_name) or "").strip()
    if env_key:
        return env_key, "environment variable"

    settings_key = (get_setting(db_name) or "").strip()
    if settings_key:
        return settings_key, "Settings"
    return None, None


def _get_api_key(provider: str) -> Optional[str]:
    """Retrieve a trimmed key from environment or SQLite without hardcoded values."""
    return _resolve_api_key(provider)[0]


def _call_openai(messages: list[dict], api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _call_groq(messages: list[dict], api_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    candidates = _get_groq_model_candidates()
    for model in candidates:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=400,
                temperature=0.4,
            )
        except Exception as error:
            if getattr(error, "status_code", None) == 404 and model != candidates[-1]:
                continue
            raise

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")
        return content.strip()

    raise RuntimeError("No Groq chat model candidates were available.")


def _call_google(messages: list[dict], api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    # Convert messages to a single prompt for Gemini
    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    response = model.generate_content(prompt)
    return response.text.strip()


def _analytical_fallback(question: str, df: pd.DataFrame) -> str:
    """
    Rule-based fallback that answers common questions analytically.
    Used when no LLM API key is available.
    """
    q = question.lower()
    kpis    = compute_kpis(df)
    cat_exp = get_category_expense(df, top_n=5)
    monthly = get_monthly_income_expense(df)

    if any(w in q for w in ["most", "highest", "top", "maximum"]):
        if not cat_exp.empty:
            top = cat_exp.iloc[0]
            return (
                f"Your highest expense category is **{top['category']}** with ₹{top['total']:,.0f} "
                f"({top['pct']}% of total expenses)."
            )

    if any(w in q for w in ["savings", "saved", "save"]):
        return (
            f"Your total savings are **₹{kpis['total_savings']:,.0f}** "
            f"with a savings rate of **{kpis['savings_rate']}%**."
        )

    if any(w in q for w in ["income", "earn", "salary"]):
        return f"Your total income is **₹{kpis['total_income']:,.0f}**."

    if any(w in q for w in ["expense", "spent", "spending", "spend"]):
        if not cat_exp.empty:
            cats = ", ".join([f"{r['category']} (₹{r['total']:,.0f})" for _, r in cat_exp.head(3).iterrows()])
            return (
                f"Your total expenses are **₹{kpis['total_expense']:,.0f}**. "
                f"Top categories: {cats}."
            )

    if any(w in q for w in ["month", "monthly", "compare", "last"]):
        if len(monthly) >= 2:
            m1, m2 = monthly.iloc[-1], monthly.iloc[-2]
            return (
                f"**{m1['month_str']}**: Expenses ₹{m1['expense']:,.0f} | "
                f"**{m2['month_str']}**: Expenses ₹{m2['expense']:,.0f}. "
                f"Change: {((m1['expense']-m2['expense'])/m2['expense']*100):+.1f}%."
            )

    if any(w in q for w in ["reduce", "cut", "save more", "budget", "tips", "suggestion"]):
        tips = generate_budget_suggestions(df)
        if tips:
            return "Here are some personalized suggestions:\n" + "\n".join(f"• {t}" for t in tips[:3])

    return (
        f"Based on your data: Total income ₹{kpis['total_income']:,.0f} | "
        f"Total expenses ₹{kpis['total_expense']:,.0f} | "
        f"Savings rate {kpis['savings_rate']}%. "
        f"💡 *Configure an LLM API key in Settings for detailed AI answers.*"
    )


def _call_provider(provider: str, messages: list[dict], api_key: str) -> str:
    """Dispatch a chat request to the explicitly selected provider."""
    if provider == "openai":
        return _call_openai(messages, api_key)
    if provider == "groq":
        return _call_groq(messages, api_key)
    if provider == "google":
        return _call_google(messages, api_key)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _safe_provider_error(provider: str, error: Exception) -> str:
    """Create a useful provider error without exposing request or credential details."""
    if isinstance(error, ModuleNotFoundError):
        package_by_provider = {"groq": "groq", "openai": "openai", "google": "google-generativeai"}
        package = package_by_provider.get(provider, provider)
        return (
            f"The {provider.title()} SDK is missing from the Python environment running this app "
            f"(missing module: {error.name or package}). Install project dependencies with "
            "the same Python interpreter (`python -m pip install -r requirements.txt`) and restart Streamlit."
        )

    status_code = getattr(error, "status_code", None)
    if status_code == 401:
        return (
            f"{provider.title()} rejected the configured API key (HTTP 401). "
            "It may be invalid, revoked, or issued for a different account. Replace it with a newly generated key."
        )
    if status_code == 403:
        return f"{provider.title()} denied this request (HTTP 403). Check account access and model permissions."
    if status_code == 404:
        if provider == "groq":
            return (
                "Groq returned HTTP 404 for every configured chat model candidate. "
                "Check the model IDs enabled for your Groq account or set GROQ_MODEL to an available chat model."
            )
        return f"{provider.title()} could not find the requested model or endpoint (HTTP 404). Check the model configuration."
    if status_code == 429:
        return f"{provider.title()} rate limit or quota reached (HTTP 429). Try again later or check your account quota."
    if isinstance(status_code, int) and status_code >= 500:
        return f"{provider.title()} is temporarily unavailable (HTTP {status_code}). Try again later."
    if type(error).__name__ in {"APIConnectionError", "ConnectError", "ConnectionError", "TimeoutError", "APITimeoutError"}:
        return f"Could not reach {provider.title()} due to a network or timeout error. Check connectivity and retry."

    return f"{provider.title()} request failed ({type(error).__name__}). Check provider settings and try again."


def test_llm_connection() -> tuple[bool, str]:
    """Perform a minimal authenticated request for the selected provider."""
    configured = (get_setting("llm_provider", "") or os.getenv("LLM_PROVIDER", "")).strip().lower()
    provider = _get_llm_provider()
    if provider is None:
        if configured in {"openai", "google", "groq"}:
            return False, f"{configured.title()} is selected, but no API key is configured for that provider."
        return False, "No LLM provider API key is configured. Analytical mode remains available."

    api_key, key_source = _resolve_api_key(provider)
    if not api_key:
        return False, f"No API key is configured for {provider.title()}."

    messages = [
        {"role": "system", "content": "Reply with the single word OK."},
        {"role": "user", "content": "Connection test. Reply OK."},
    ]
    try:
        _call_provider(provider, messages, api_key)
    except Exception as error:
        message = _safe_provider_error(provider, error)
        if getattr(error, "status_code", None) == 401 and key_source:
            message += f" The rejected key came from {key_source}; update that source (environment variables take precedence over Settings)."
        return False, message
    return True, f"{provider.title()} API request succeeded."


def ask_assistant(question: str, df: pd.DataFrame, chat_history: list[dict]) -> tuple[str, str]:
    """
    Main entry point for the AI assistant.
    Returns: (answer_text, provider_used)
    """
    if df.empty:
        return (
            "Please upload your transaction data first so I can analyze your finances! "
            "You can use the **Transactions** page to upload a CSV file.",
            "none",
        )

    context  = _build_financial_context(df)
    provider = _get_llm_provider()

    if provider is None:
        selected = (get_setting("llm_provider", "") or os.getenv("LLM_PROVIDER", "")).strip().lower()
        if selected in {"openai", "google", "groq"}:
            answer = (
                f"{selected.title()} is selected, but its API key is not configured. "
                "Using analytical mode instead.\n\n"
                + _analytical_fallback(question, df)
            )
            return answer, "analytical (missing selected-provider key)"
        answer = _analytical_fallback(question, df)
        return answer, "analytical"

    system_msg = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [{"role": "system", "content": system_msg}]

    # Add recent chat history (last 6 turns)
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    api_key, key_source = _resolve_api_key(provider)
    if not api_key:
        answer = _analytical_fallback(question, df)
        return answer, "analytical"

    try:
        answer = _call_provider(provider, messages, api_key)
    except Exception as e:
        answer = (
            f"{_safe_provider_error(provider, e)}"
            + (f" Key source: {key_source}." if getattr(e, "status_code", None) == 401 and key_source else "")
            + " Switching to analytical mode.\n\n"
            + _analytical_fallback(question, df)
        )
        provider = "analytical (fallback)"

    return answer, provider


def get_quick_questions() -> list[str]:
    """Return suggested quick questions for the assistant."""
    return [
        "Where did I spend the most money this month?",
        "What is my total savings?",
        "How can I reduce unnecessary expenses?",
        "Compare my expenses for the last two months.",
        "Which category has the highest spending?",
        "What are my major spending patterns?",
        "How much did I spend on Food & Dining?",
        "What is my average monthly spending?",
    ]
