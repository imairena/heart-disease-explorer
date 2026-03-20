import streamlit as st
import pandas as pd
import html

from ai_service import ask_ai, get_gemini_api_key


FOCUS_OPTIONS = ["General", "Overview & Visualizations", "What-If Analysis", "Data Summary"]


def _format_what_if_context() -> str:
    what_if = st.session_state.get("what_if_context")
    if not what_if:
        return "What-If context: No recent What-If interaction yet."

    f = what_if["features"]
    d = what_if["display_values"]
    return (
        "What-If context:\n"
        f"- Prediction model: {what_if['prediction_model']}\n"
        f"- Estimated risk: {what_if['risk_pct']:.1f}%\n"
        f"- Age: {f['age']} | Sex: {d['sex']}\n"
        f"- Blood Pressure: {f['trestbps']} mm Hg | Cholesterol: {f['chol']} mg/dl\n"
        f"- Fasting Blood Sugar > 120: {d['fbs']} | Exercise Angina: {d['exang']}\n"
        f"- Chest Pain: {d['cp']} | Max Heart Rate: {f['thalach']}\n"
        f"- Rest ECG: {d['restecg']} | ST Depression: {f['oldpeak']} | ST Slope: {d['slope']}\n"
        f"- Major vessels: {f['ca']} | Thalassemia: {d['thal']}"
    )


def _build_context(df: pd.DataFrame, focus: str) -> str:
    disease_pct = df["num"].mean() * 100
    numeric = [c for c in df.columns if c != "num"]
    corr = df[numeric + ["num"]].corr(numeric_only=True)["num"].drop("num").sort_values(key=abs, ascending=False)
    top_corr = corr.head(5)
    corr_text = ", ".join([f"{k} ({v:+.2f})" for k, v in top_corr.items()])

    base = (
        "App: Heart Disease Explorer (educational).\n"
        f"Dataset rows: {len(df)}\n"
        f"Disease prevalence: {disease_pct:.1f}%\n"
        f"Top absolute correlations with disease target: {corr_text}\n"
    )

    if focus == "Overview & Visualizations":
        focus_text = (
            "User focus: Visualizations.\n"
            "The page includes correlation heatmap, feature distributions by disease status, "
            "target/age breakdown, and risk factor box plots."
        )
    elif focus == "What-If Analysis":
        focus_text = "User focus: What-If Analysis.\n" + _format_what_if_context()
    elif focus == "Data Summary":
        desc = df.describe().transpose()
        top_std = desc["std"].sort_values(ascending=False).head(4)
        std_text = ", ".join([f"{idx} (std={val:.2f})" for idx, val in top_std.items()])
        focus_text = (
            "User focus: Data Summary.\n"
            f"Highest-variance features: {std_text}.\n"
            "User can inspect sample rows and descriptive statistics."
        )
    else:
        focus_text = "User focus: General across all app tabs.\n" + _format_what_if_context()

    return f"{base}\n{focus_text}"


def render_global_copilot(df: pd.DataFrame):
    """Render a global AI copilot available across the app."""
    st.markdown(
        """
<style>
    .st-key-open_copilot_fab {
        position: fixed;
        right: 50px;
        bottom: 50px;
        z-index: 10001;
    }
    .st-key-open_copilot_fab button {
        width: 75px;
        height: 75px;
        border-radius: 50%;
        font-size: 3.6rem;
        border: 2px solid #E06D53;
        background: #FF8A66 !important;
        color: #111111 !important;
        box-shadow: 0 0 0 4px rgba(255, 138, 102, 0.3), 0 10px 24px rgba(0, 0, 0, 0.35);
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .st-key-open_copilot_fab button p,
    .st-key-open_copilot_fab button div,
    .st-key-open_copilot_fab button span {
        font-size: 3.6rem !important; 
    }
    .st-key-copilot_panel {
        position: fixed;
        right: 22px;
        bottom: 128px;
        width: min(460px, calc(100vw - 36px));
        max-height: 72vh;
        overflow-y: auto;
        z-index: 10000;
        background: #FCFCFC;
        border: 2px solid #E6E6E6;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.28);
    }
    .st-key-copilot_panel h4, .st-key-copilot_panel h2, .st-key-copilot_panel h3, .st-key-copilot_panel p, .st-key-copilot_panel label, .st-key-copilot_panel div, .st-key-copilot_panel span {
        color: #111111 !important;
    }
    
    .copilot-msg-user {
        background: #FFE9E2;
        border: 1px solid #F6C2B4;
        color: #111111;
        border-radius: 10px;
        padding: 8px 10px;
        margin: 8px 0;
    }
    .copilot-msg-assistant {
        background: #FFFFFF;
        border: 1px solid #DDDDDD;
        color: #111111;
        border-radius: 10px;
        padding: 8px 10px;
        margin: 8px 0;
    }
    /* Conversation thread: light grey shell so chat is distinct from controls */
    .copilot-conversation-box {
        background: #EEEEEF;
        border: 1px solid #E2E2E5;
        border-radius: 12px;
        padding: 12px;
        margin: 10px 0;
        max-height: 38vh;
        overflow-y: auto;
    }
    .st-key-copilot_close_btn button,
    .st-key-copilot_send_btn button,
    .st-key-copilot_clear_btn button {
        background-color: #E06D53 !important;
        color: #111111 !important;
        border: none !important;
    }
    .st-key-copilot_close_btn button:hover,
    .st-key-copilot_send_btn button:hover,
    .st-key-copilot_clear_btn button:hover {
        background-color: #FF8A66 !important;
    }
    .st-key-copilot_panel .stButton > button:hover {
        border-color: #E06D53 !important;
        box-shadow: 0 0 0 2px rgba(224, 109, 83, 0.25) !important;
    }
    .st-key-copilot_panel [data-baseweb="select"] {
        background: #112240 !important;
        border-radius: 8px !important;
    }
    .st-key-copilot_panel [data-baseweb="select"] * {
        color: #FFFFFF !important;
    }
    .st-key-copilot_panel [role="option"] {
        color: #FFFFFF !important;
        background: #112240 !important;
    }
    .st-key-copilot_panel [role="option"]:hover {
        background: #E06D53 !important;
        color: #111111 !important;
    }
    /* Message field: white background; typed text black; placeholder stays muted grey */
    .st-key-copilot_panel .stTextInput [data-baseweb="input"],
    .st-key-copilot_panel .stTextInput [data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        border-color: #CFCFCF !important;
    }
    .st-key-copilot_panel .stTextInput input {
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        caret-color: #E06D53 !important;
    }
    .st-key-copilot_panel .stTextInput input::placeholder {
        color: #8892b0 !important;
        -webkit-text-fill-color: #8892b0 !important;
    }
    /* Hide Streamlit "Press Enter to apply" hint under the text field */
    .st-key-copilot_panel [data-testid="InputInstructions"] {
        display: none !important;
    }
</style>
        """,
        unsafe_allow_html=True,
    )

    if "global_messages" not in st.session_state:
        st.session_state.global_messages = []
    if "copilot_focus" not in st.session_state:
        st.session_state.copilot_focus = "General"
    if "copilot_open" not in st.session_state:
        st.session_state.copilot_open = False
    if "copilot_input_nonce" not in st.session_state:
        st.session_state.copilot_input_nonce = 0

    if st.button("🤖", key="open_copilot_fab", help="Open AI Copilot"):
        st.session_state.copilot_open = True

    if not st.session_state.copilot_open:
        return

    panel = st.container(key="copilot_panel")
    with panel:
        if not st.session_state.global_messages:
            st.session_state.global_messages.append(
                {
                    "role": "assistant",
                    "content": "Hello, I am your AI assistant. Feel free to ask questions about visualizations, trends, risk factors, and your What-If profile.",
                }
            )

        top_cols = st.columns([5, 1])
        with top_cols[0]:
            st.markdown("#### AI Copilot")
        with top_cols[1]:
            if st.button("✕", key="copilot_close_btn", help="Close panel"):
                st.session_state.copilot_open = False
                st.rerun()

        st.caption("Ask about risks, visualizations, trends, and your What-If profile.")

        st.session_state.copilot_focus = st.selectbox(
            "Copilot focus",
            FOCUS_OPTIONS,
            index=FOCUS_OPTIONS.index(st.session_state.copilot_focus),
            help="Choose what context the AI should prioritize for this question.",
        )

        clear_col, _ = st.columns([1, 4])
        if clear_col.button("New chat", key="copilot_clear_btn", help="Clear conversation"):
            st.session_state.global_messages = [
                {
                    "role": "assistant",
                    "content": "Hello, I am your AI assistant. Feel free to ask questions about visualizations, trends, risk factors, and your What-If profile.",
                }
            ]

        conv_parts = []
        for message in st.session_state.global_messages:
            cls = "copilot-msg-user" if message["role"] == "user" else "copilot-msg-assistant"
            safe_content = html.escape(message["content"]).replace("\n", "<br>")
            conv_parts.append(f'<div class="{cls}">{safe_content}</div>')
        st.markdown(
            f'<div class="copilot-conversation-box">{"".join(conv_parts)}</div>',
            unsafe_allow_html=True,
        )

        input_col, send_col = st.columns([4, 1])
        with input_col:
            # New key after each successful send so the field resets without touching
            # session_state after the widget is instantiated (Streamlit restriction).
            _input_key = f"global_copilot_input_text_{st.session_state.copilot_input_nonce}"
            typed_prompt = st.text_input(
                "Message",
                value="",
                placeholder="Ask about visualizations, risk, trends, or prevention ideas...",
                key=_input_key,
                label_visibility="collapsed",
            )
        with send_col:
            send_clicked = st.button("Send", key="copilot_send_btn", use_container_width=True)

        prompt = typed_prompt.strip() if send_clicked else ""

        if prompt:
            if not get_gemini_api_key():
                st.error("API key missing. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml`.")
                return

            st.session_state.global_messages.append({"role": "user", "content": prompt})

            try:
                context = _build_context(df, st.session_state.copilot_focus)
                with st.spinner("Thinking..."):
                    answer = ask_ai(prompt, context, st.session_state.global_messages)
                st.session_state.global_messages.append({"role": "assistant", "content": answer})
                st.session_state.copilot_input_nonce += 1
            except Exception as exc:
                st.error(f"AI error: {exc}")
