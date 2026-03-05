import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

from utils.helpers import encode_image_to_base64, load_mock_dwh, format_currency
from src.extraction import extract_bill_data
from src.dwh_matcher import match_customer
from src.rag_engine import generate_prompt_package
from src.vector_store import retrieve_knowledge
from src.generator import generate_final_answer

load_dotenv()

st.set_page_config(page_title="PPC AI Billing Agent", page_icon="⚡", layout="wide")

@st.cache_data(show_spinner=False)
def run_cached_extraction(image_bytes):
    base64_image = encode_image_to_base64(image_bytes)
    image_uri = f"data:image/jpeg;base64,{base64_image}"
    return extract_bill_data(image_uri)

st.markdown("""
<style>
    .title-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffebee 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: -50px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .main-title {
        color: #00b4d8; 
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-title {
        color: #ff5252; 
        font-size: 1.5rem; 
        font-weight: 500;
        margin-top: 0.5rem;
    }
    [data-testid="stMetric"] {
        background-color: #f1f8ff; 
        border: 1px solid #cce5ff;
        border-left: 6px solid #00b4d8; 
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricValue"] {
        color: #ff5252; 
    }
</style>
""", unsafe_allow_html=True)

def reset_session():
    st.session_state.extracted_data = None
    st.session_state.dwh_result = None
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "dwh_result" not in st.session_state:
    st.session_state.dwh_result = None

st.markdown("""
<div class="title-box">
    <h1 class="main-title">⚡ PPC AI Billing Assistant</h1>
    <p class="sub-title">Upload your electricity bill to get structured insights and ask questions grounded in company policy.</p>
</div>
""", unsafe_allow_html=True)

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("<h3 style='color: #00b4d8;'>📄 Upload Bill</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your bill (PNG, JPG)", 
        type=["png", "jpg", "jpeg"],
        on_change=reset_session 
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a bill image to begin the analysis.")
        
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Bill Document", width="stretch")
        
        # Adding an else block to maintain the UI tree structure prevents the "greyed out" ghosting bug during reruns
        if st.session_state.extracted_data is None:
            with st.status("⚙️ Processing Billing Document...", expanded=True) as status:
                try:
                    st.write("🔍 Running Vision OCR...")
                    st.session_state.extracted_data = run_cached_extraction(uploaded_file.getvalue())
                    
                    st.write("🗄️ Querying Data Warehouse...")
                    mock_dwh = load_mock_dwh()
                    st.session_state.dwh_result = match_customer(st.session_state.extracted_data.customer_id, mock_dwh)
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Processing Failed", state="error", expanded=True)
                    st.error(f"Extraction failed: {e}")
        else:
            with st.status("✅ Analysis Complete! (Loaded from cache)", state="complete", expanded=False):
                st.write("Data retrieved from previous extraction.")

    if st.session_state.extracted_data and st.session_state.dwh_result:
        data = st.session_state.extracted_data
        
        st.divider()
        st.markdown("<h3 style='color: #ff5252;'>🔍 Bill Summary</h3>", unsafe_allow_html=True)
        with st.container():
            summary_df = pd.DataFrame({
                "Detail": ["Customer ID", "Total Amount", "Period", "Tariff"],
                "Value": [data.customer_id, format_currency(data.total_amount), data.billing_period, data.tariff_code]
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            
            with st.expander("📊 View Further Details"):
                for key, value in data.model_dump().items():
                    if isinstance(value, list):
                        st.markdown(f"**{str(key).replace('_', ' ').title()}:**")
                        if len(value) > 0 and isinstance(value[0], dict):
                            st.dataframe(pd.DataFrame(value), hide_index=True, use_container_width=True)
                        else:
                            for item in value:
                                st.write(f"- {item}")
                    else:
                        st.markdown(f"**{str(key).replace('_', ' ').title()}:** {value}")
        
        dwh_status = st.session_state.dwh_result['status'] 
        if dwh_status == "single_match":
            st.success("✅ **DWH Match:** Customer found. Context loaded securely.", icon="🔐")
        elif dwh_status == "multiple_matches":
            st.warning("⚠️ **DWH Match:** Multiple accounts found for this ID. Clarification needed.", icon="⚠️")
        else:
            st.error("❌ **DWH Match:** No matching customer found in database.", icon="❌")

with col_right:
    if st.session_state.extracted_data and st.session_state.dwh_result:
        data = st.session_state.extracted_data
        dwh_status = st.session_state.dwh_result['status']

        st.markdown("<h3 style='color: #00b4d8;'>💬 Ask the AI Agent</h3>", unsafe_allow_html=True)
        
        with st.form(key="chat_input_form", clear_on_submit=True):
            cols = st.columns([5, 1])
            with cols[0]:
                prompt = st.text_input(
                    "Ask a question:", 
                    label_visibility="collapsed", 
                    placeholder="E.g., Why is this bill higher than last month?"
                )
            with cols[1]:
                submit_button = st.form_submit_button("Send")

        chat_container = st.container(height=520, border=True)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if submit_button and prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing rules and history..."):
                        
                        pkg = generate_prompt_package(
                            user_query=prompt,
                            dwh_status=dwh_status,
                            extracted_data=data.model_dump_json()
                        )
                        
                        with st.expander("⚙️ System: View RAG Prompt Package & Confidence"):
                            # Added the numeric value alongside the confidence score text
                            st.caption(f"Routing Confidence Score: **{int(pkg.confidence_score * 100)}%**")
                            st.progress(pkg.confidence_score)
                            st.write("**Generated Retrieval Queries:**", pkg.retrieval_queries)
                            st.write(f"**Routing Status:** {dwh_status}")
                        
                        retrieved_docs = retrieve_knowledge(pkg.retrieval_queries)
                        with st.expander("📚 View Retrieved Knowledge (ChromaDB)"):
                            if retrieved_docs:
                                st.markdown(retrieved_docs)
                            else:
                                st.write("No documents retrieved (Database might be empty).")
                                
                        if dwh_status == "single_match" or dwh_status == "no_match":
                            response = generate_final_answer(
                                system_instructions=pkg.system_instructions,
                                customer_data=st.session_state.dwh_result['data'] if dwh_status == "single_match" else "No Data",
                                retrieved_docs=retrieved_docs,
                                user_query=prompt
                            )
                        else:
                            response = f"**I need clarification:** {pkg.clarifying_questions[0]}"
                        
                        st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})