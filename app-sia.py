import os
import streamlit as st
from main import run_pipeline

st.set_page_config(
    page_title="RAGnarok Billing Agent",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(168,85,247,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(34,211,238,0.14), transparent 24%),
            linear-gradient(135deg, #020617 0%, #0f172a 45%, #1e1b4b 100%);
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(232,121,249,0.25);
        background: rgba(232,121,249,0.10);
        color: #f5d0fe;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0 0 0.6rem 0;
        background: linear-gradient(90deg, #ffffff 0%, #f5d0fe 45%, #bae6fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 850px;
        margin-bottom: 1rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        backdrop-filter: blur(12px);
        border-radius: 22px;
        padding: 1.2rem 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .metric-tile {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 0.9rem 1rem;
    }

    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #cbd5e1;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
    }

    .step {
        border-radius: 18px;
        padding: 0.9rem 1rem;
        text-align: center;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
        color: #dbe4f0;
    }

    .step.active {
        background: rgba(34,211,238,0.12);
        color: #cffafe;
        border: 1px solid rgba(34,211,238,0.30);
    }

    .step.done {
        background: rgba(52,211,153,0.12);
        color: #d1fae5;
        border: 1px solid rgba(52,211,153,0.28);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.35rem;
    }

    .section-subtitle {
        color: #dbe4f0;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .doc-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .doc-title {
        font-weight: 700;
        color: white;
        margin-bottom: 0.35rem;
    }

    .doc-citation {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: rgba(34,211,238,0.12);
        color: #bae6fd;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .answer-box {
        background: linear-gradient(90deg, rgba(217,70,239,0.12), rgba(34,211,238,0.10));
        border: 1px solid rgba(217,70,239,0.16);
        border-radius: 20px;
        padding: 1.1rem 1rem;
        color: #f8fafc;
        line-height: 1.8;
        font-size: 0.98rem;
    }

    .small-muted {
        color: #cbd5e1;
        font-size: 0.85rem;
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #d946ef, #22d3ee);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    div[data-testid="stFileUploader"] button:hover {
        opacity: 0.9;
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.04);
        border: 1px dashed rgba(255,255,255,0.18);
        border-radius: 20px;
        padding: 0.8rem;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] label {
        color: #dbe4f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-badge">RAGnarok</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">RAGnarok Billing Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'An elegant multimodal copilot for electricity billing: upload a bill, extract structured fields, '
    'match the customer in the DWH, retrieve grounded knowledge, and generate a transparent answer with citations.'
    "</div>",
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([3, 1])
with top_right:
    st.markdown(
        """
        <div class="glass-card">
            <div class="metric-label">Team</div>
            <div class="metric-value">RAGnarok</div>
            <div class="small-muted">Azure OpenAI + Azure AI Search</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader("Upload electricity bill", type=["png", "jpg", "jpeg"])

has_file = uploaded_file is not None
steps = [
    ("Upload Bill", "done" if has_file else "active"),
    ("Vision Extraction", "done" if has_file else "idle"),
    ("DWH Matching", "done" if has_file else "idle"),
    ("RAG Retrieval", "done" if has_file else "idle"),
    ("Grounded Answer", "done" if has_file else "idle"),
]

step_cols = st.columns(5)
for i, (label, state) in enumerate(steps):
    with step_cols[i]:
        st.markdown(
            f'<div class="step {state}">{label}</div>',
            unsafe_allow_html=True,
        )

st.write("")

def safe_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

if uploaded_file:
    temp_path = "temp_bill.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    left, right = st.columns([1.05, 0.95])

    with st.spinner("RAGnarok is analyzing the bill..."):
        user_q = "Why is my bill higher?"
        mock_dwh = None
        chat_history = []

        try:
            result = run_pipeline(temp_path, user_q, mock_dwh, chat_history)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            result = None

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">1. Bill Upload</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Uploaded bill image ready for analysis.</div>',
            unsafe_allow_html=True,
        )
        st.image(uploaded_file, use_container_width=True)
        st.caption(uploaded_file.name)
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">2. Extracted Bill Data</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Structured fields generated by the vision model.</div>',
            unsafe_allow_html=True,
        )

        if result:
            extracted = safe_get(result, "extracted_data", {})
            confidence = safe_get(extracted, "extraction_confidence", None)
            if confidence is None:
                confidence = safe_get(extracted, "confidence_score", 0)

            st.markdown(
                f"""
                <div class="metric-tile" style="margin-bottom: 1rem;">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{round(float(confidence) * 100) if confidence is not None else 0}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-label">Customer ID</div>
                        <div class="metric-value">{safe_get(extracted, "customer_id", "Unknown")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="metric-tile" style="margin-top: 0.8rem;">
                        <div class="metric-label">Billing Period</div>
                        <div class="metric-value">{safe_get(extracted, "billing_period", "Unknown")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="metric-tile" style="margin-top: 0.8rem;">
                        <div class="metric-label">Consumption</div>
                        <div class="metric-value">{safe_get(extracted, "consumption_kwh", "Unknown")} kWh</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-label">Account</div>
                        <div class="metric-value">{safe_get(extracted, "contract_account_num", "Unknown")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="metric-tile" style="margin-top: 0.8rem;">
                        <div class="metric-label">Tariff</div>
                        <div class="metric-value">{safe_get(extracted, "tariff_code", "Unknown")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="metric-tile" style="margin-top: 0.8rem;">
                        <div class="metric-label">Total Amount</div>
                        <div class="metric-value">€{safe_get(extracted, "total_amount", "Unknown")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            summary = safe_get(extracted, "bill_summary", safe_get(extracted, "summary", "No summary available."))
            st.write("")
            st.markdown("**Summary**")
            st.write(summary)

            st.write("")
            st.markdown("**Line Items**")
            line_items = safe_get(extracted, "line_items", [])
            if line_items:
                rows = []
                for item in line_items:
                    rows.append(
                        {
                            "Description": safe_get(item, "description", ""),
                            "Type": safe_get(item, "charge_type", ""),
                            "Amount": safe_get(item, "amount", ""),
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No line items extracted.")
        else:
            st.info("No extraction output available.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">3. Customer Match</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Profile enriched from the billing warehouse.</div>',
            unsafe_allow_html=True,
        )

        if result:
            customer_match = safe_get(result, "customer_match", {})
            st.json(customer_match)
        else:
            st.info("No customer match available.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">4. Retrieved Knowledge</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Top chunks returned from the retrieval layer.</div>',
            unsafe_allow_html=True,
        )

        if result:
            retrieved_docs = safe_get(result, "retrieved_docs", [])
            if isinstance(retrieved_docs, str):
                st.write(retrieved_docs)
            elif retrieved_docs:
                for doc in retrieved_docs:
                    title = safe_get(doc, "title", "Knowledge Chunk")
                    snippet = safe_get(doc, "snippet", safe_get(doc, "page_content", ""))
                    citation = safe_get(doc, "citation", "[Source]")
                    st.markdown(
                        f"""
                        <div class="doc-card">
                            <div class="doc-title">{title}</div>
                            <div>{snippet}</div>
                            <div class="doc-citation">{citation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No retrieved documents available.")
        else:
            st.info("No retrieval output available.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">5. Grounded Answer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Personalized answer with supporting citations.</div>',
            unsafe_allow_html=True,
        )

        if result:
            final_answer = safe_get(result, "final_answer", "No final answer available.")
            st.markdown(
                f'<div class="answer-box">{final_answer}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No grounded answer available.")

        st.markdown("</div>", unsafe_allow_html=True)

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

else:
    st.markdown(
        """
        <div class="glass-card" style="margin-top: 1rem;">
            <div class="section-title">Ready for Demo</div>
            <div class="section-subtitle">
                Upload a bill image to launch the full RAGnarok pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )