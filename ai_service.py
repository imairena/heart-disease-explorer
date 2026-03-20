import streamlit as st
import google.generativeai as genai


def get_gemini_api_key() -> str:
    """Return Gemini API key from Streamlit secrets, or empty string."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        return ""


def ask_ai(user_prompt: str, context: str, chat_history: list[dict]) -> str:
    """Send prompt + app context to Gemini and return a response."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .streamlit/secrets.toml")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    recent_history = chat_history[-8:] if chat_history else []
    history_text = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in recent_history if "role" in m and "content" in m]
    )

    full_prompt = f"""
You are a helpful medical education copilot in a Heart Disease Explorer app.

Rules:
- Be clear, concise, and empathetic.
- Focus on explanation and risk-awareness guidance.
- Never diagnose, never prescribe medication, and never claim certainty.
- Mention uncertainty when needed.
- End with a short disclaimer: "I am an AI assistant, not a doctor."

App context:
{context}

Recent chat:
{history_text if history_text else "(No prior messages)"}

User question:
{user_prompt}
""".strip()

    response = model.generate_content(full_prompt)
    answer = (response.text or "").strip()

    if "not a doctor" not in answer.lower():
        answer += "\n\nI am an AI assistant, not a doctor."
    return answer
