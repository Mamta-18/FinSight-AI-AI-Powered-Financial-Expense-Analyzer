"""
pages/assistant.py
MODULE 6 – AI Financial Assistant chatbot with LLM + analytical fallback.
"""

import streamlit as st
import pandas as pd
from modules.ai_assistant import ask_assistant, get_quick_questions, DISCLAIMER, _get_llm_provider


def show(df: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🤖 AI Financial Assistant</h1>
        <p class="page-subtitle">Ask questions about your spending, savings, and get personalized insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # Privacy note
    st.markdown("""
    <div class="insight-card insight-info" style="font-size:0.82rem;">
        🔒 <b>Privacy:</b> Only aggregated financial summaries (category totals, monthly sums) are sent
        to the LLM API — not individual transaction descriptions. Configure your API key in
        <b>Settings</b> for AI-powered responses. Without a key, analytical mode is used automatically.
    </div>
    """, unsafe_allow_html=True)

    # Provider status
    provider = _get_llm_provider()
    if provider:
        st.markdown(f"""
        <div class="insight-card insight-success" style="font-size:0.82rem;">
            ✅ Selected provider: <b>{provider}</b> — API availability is verified when you send a message.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-card insight-warning" style="font-size:0.82rem;">
            ⚡ No API key configured — running in <b>analytical mode</b>.
            Add an API key in <b>Settings → AI Configuration</b> for GPT/Gemini/Groq responses.
        </div>
        """, unsafe_allow_html=True)

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Greeting
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "👋 Hi! I'm FinSight AI, your personal financial assistant. "
                "I can help you understand your spending patterns, savings, and give budget tips. "
                "What would you like to know about your finances?"
            ),
        })

    # ── Quick Questions ────────────────────────────────────────────
    st.markdown("#### 💬 Quick Questions")
    quick = get_quick_questions()
    cols  = st.columns(4)
    for i, q in enumerate(quick[:8]):
        if cols[i % 4].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("⏳ Thinking..."):
                answer, prov = ask_assistant(q, df, st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.markdown("---")

    # ── Chat Display ───────────────────────────────────────────────
    st.markdown("#### 💬 Chat")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # ── Input ──────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a financial question...",
            placeholder="e.g. Where did I spend the most money?",
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([5, 1])
        with col_send:
            submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)
        with col_clear:
            clear = st.form_submit_button("🗑️ Clear", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("⏳ Analyzing your finances..."):
            answer, prov = ask_assistant(
                user_input.strip(), df, st.session_state.chat_history[:-1]
            )
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if clear:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Chat cleared. How can I help you with your finances?",
        }]
        st.rerun()

    # Disclaimer
    st.markdown(f"""
    <div style="color:#64748b;font-size:0.75rem;margin-top:16px;text-align:center;">{DISCLAIMER}</div>
    """, unsafe_allow_html=True)
