import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
print("Tracing enabled:", os.environ.get("LANGCHAIN_TRACING_V2"))

import streamlit as st
from utils.helpers import encode_image_to_base64, load_mock_dwh, format_currency
from src.extraction import extract_bill_data
from src.dwh_matcher import match_customer
from src.rag_engine import generate_prompt_package
from src.vector_store import retrieve_knowledge
from src.generator import generate_final_answer

import io
from openai import AzureOpenAI
from streamlit_mic_recorder import mic_recorder


st.set_page_config(page_title="PPC AI Billing Agent", page_icon="⚡", layout="wide")

@st.cache_data(show_spinner=False)
def run_cached_extraction(image_bytes):
    base64_image = encode_image_to_base64(image_bytes)
    image_uri = f"data:image/jpeg;base64,{base64_image}"
    return extract_bill_data(image_uri)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }

    /* Entire App Background */
    .stApp {
        background-color: #F4F8FB;
        background-image: radial-gradient(circle at top right, #E8F2FB 0%, #F4F8FB 100%);
    }
            
    [data-testid="stHeader"] {
        visibility: hidden;
        height: 0%;
    }

    /* Hide the default Streamlit footer */
    [data-testid="stFooter"] {
        visibility: hidden;
    }

    /* Hide the toolbar  */
    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    /* Vibrant Header Box */
    .title-box {
        background: linear-gradient(135deg, #005BAA 0%, #00C3D9 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        margin-top: -60px; 
        box-shadow: 0 10px 25px rgba(0, 91, 170, 0.2);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .main-title {
        color: #FFFFFF !important; 
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
        letter-spacing: -0.5px;
        text-shadow: 1px 2px 4px rgba(0,0,0,0.15);
    }
    .sub-title {
        color: #E0F7FA !important; 
        font-size: 1.3rem; 
        font-weight: 500;
        margin-top: 0.5rem;
    }

    /* Interactive Metric Cards with Hover Effects */
    [data-testid="stMetric"] {
        background-color: #FFFFFF; 
        border: 1px solid #E2E8F0;
        border-left: 6px solid #00C3D9; 
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px rgba(0, 195, 217, 0.15);
        border-color: #00C3D9;
    }
    [data-testid="stMetricValue"] {
        color: #005BAA; 
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    /* Chat Bubble Styling (SaaS Look) */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #E9ECEF;
    }
    
    /* Clean Dataframes & Expanders */
    th {
        background-color: #E8F2FB !important;
        color: #005BAA !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
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

if not os.getenv("AZURE_OPENAI_API_KEY"):
    st.error("AZURE_OPENAI_API_KEY not found")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("<h3 style='color: #005BAA; font-weight: 800;'>📄 Upload Bill</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your bill (PNG, JPG)", 
        type=["png", "jpg", "jpeg"],
        on_change=reset_session 
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a bill image to begin the analysis.")
        
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Bill Document", width="stretch")
        
        if st.session_state.extracted_data is None:
            with st.status("⚙️ Processing Billing Document...", expanded=True) as status:
                try:
                    st.write("🔍 Running Vision OCR...")
                    st.session_state.extracted_data = run_cached_extraction(uploaded_file.getvalue())
                    
                    st.write("🗄️ Querying Data Warehouse...")
                    mock_dwh = load_mock_dwh()
                    st.session_state.dwh_result = match_customer(st.session_state.extracted_data.customer_id, mock_dwh)
                    

                    dwh_status = st.session_state.dwh_result.get('status')
                    if dwh_status not in ["single_match", "multiple_matches"]:
                        st.write("No DWH match found. Generating clarification prompt...")
                        
                        # Give the LLM specific instructions for this edge case
                        system_msg = (
                            "You are a helpful AI Billing Assistant. "
                            f"We successfully extracted data from the user's uploaded bill, but the Customer ID we found ({st.session_state.extracted_data.customer_id}) "
                            "does NOT exist in our Data Warehouse database. "
                            "Politely inform the user about this issue, mention the Customer ID that was extracted, "
                            "and ask them to verify if the ID on the bill is correct, or if they have an updated account number."
                        )
                        
                        proactive_response = generate_final_answer(
                            system_instructions=system_msg,
                            customer_data="No matching records found in Data Warehouse.",
                            retrieved_docs="Not applicable. Database mismatch.",
                            user_query="My document was just processed, but there's a DWH mismatch. What should I do?",
                            chat_history=""
                        )
                        
                        st.session_state.messages.append({"role": "assistant", "content": proactive_response})
                    
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
        st.markdown("<h3 style='color: #005BAA; font-weight: 800;'>🔍 Bill Summary</h3>", unsafe_allow_html=True)
        
        confidence_pct = int(data.extraction_confidence * 100)
        st.caption(f"Extraction Confidence Score: **{confidence_pct}%**")
        st.progress(data.extraction_confidence)
        if confidence_pct < 80:
            st.warning("Low confidence extraction. Please verify the details below.")

        st.info(f"**AI Summary:** {data.bill_summary}")

        with st.container():
            summary_df = pd.DataFrame({
                "Detail": ["Customer ID", "Account Number", "Total Amount", "Period", "Tariff", "Consumption"],
                "Value": [data.customer_id, data.contract_account_num, format_currency(data.total_amount), data.billing_period, data.tariff_code, f"{data.consumption_kwh} kWh"]
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            
            st.markdown("<h4 style='color: #00C3D9; font-weight: 700; margin-top: 1rem;'>📊 Extracted Line Items</h4>", unsafe_allow_html=True)
            if data.line_items:
                lines_df = pd.DataFrame([item.model_dump() for item in data.line_items])
                lines_df['amount'] = lines_df['amount'].apply(lambda x: format_currency(x))
                lines_df.columns = [col.replace('_', ' ').title() for col in lines_df.columns]
                st.dataframe(lines_df, hide_index=True, use_container_width=True)
            else:
                st.write("No line items extracted.")
        
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

        st.markdown("<h3 style='color: #005BAA; font-weight: 800;'>💬 Ask the AI Agent</h3>", unsafe_allow_html=True)
        
        # Initialize Azure OpenAI Client
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # This will hold the final question, whether it comes from text or voice
        prompt_to_process = None

        # Layout for Voice and Text
        mic_col, input_col = st.columns([1, 5])
        
        with mic_col:
            audio = mic_recorder(
                start_prompt="🎙️ Record",
                stop_prompt="🛑 Stop",
                use_container_width=True,
                just_once=True,
                key='whisper_mic'
            )
            
            # If voice is recorded, transcribe and instantly set it to be processed
            if audio:
                with st.spinner("Transcribing..."):
                    audio_bytes = audio['bytes']
                    audio_file = io.BytesIO(audio_bytes)
                    audio_file.name = "audio.webm" 

                    transcription = client.audio.transcriptions.create(
                        model=os.getenv("AZURE_WHISPER_DEPLOYMENT"),
                        file=audio_file
                    )
                    prompt_to_process = transcription.text

        with input_col:
            with st.form(key="chat_input_form", clear_on_submit=True):
                cols = st.columns([5, 1])
                with cols[0]:
                    text_input = st.text_input(
                        "Ask a question:", 
                        label_visibility="collapsed",
                        placeholder="Type a question here..."
                    )
                with cols[1]:
                    submit_button = st.form_submit_button("Send")
                
                # If text is submitted, set it to be processed
                if submit_button and text_input:
                    prompt_to_process = text_input

        # ----------------------------------------------------------------
        # Chat Rendering & Execution
        chat_container = st.container(height=520, border=False)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # RAG Pipeline: Fires immediately if either voice or text provided a prompt
        if prompt_to_process:
            # 1. Compile history
            chat_history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
            if not chat_history_text:
                chat_history_text = "No previous conversation history."

            st.session_state.messages.append({"role": "user", "content": prompt_to_process})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt_to_process)
                    
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Searching Azure Knowledge Base and analyzing..."):
                        
                        # 2. Generate Search Queries
                        pkg = generate_prompt_package(
                            user_query=prompt_to_process,
                            dwh_status=dwh_status,
                            extracted_data=data.model_dump_json(),
                            chat_history=chat_history_text 
                        )
                        
                        # 3. Fetch from Azure AI Search
                        retrieved_docs = retrieve_knowledge(pkg.retrieval_queries)
                        
                        with st.expander("⚙️ System: View Azure Retrieval & Prompt Package"):
                            st.caption(f"Routing Confidence Score: **{int(pkg.confidence_score * 100)}%**")
                            st.progress(pkg.confidence_score)
                            st.write("**Generated Retrieval Queries:**", pkg.retrieval_queries)
                            st.markdown(f"**Retrieved Knowledge Chunks:**\n\n{retrieved_docs}")
                        
                        # 4. Smart Routing Logic
                        if pkg.clarifying_questions and len(pkg.clarifying_questions) > 0:
                            response = f"**I need clarification:** {pkg.clarifying_questions[0]}"
                        else:
                            customer_data_payload = "No Data"
                            if st.session_state.dwh_result and st.session_state.dwh_result.get('status') == "single_match":
                                customer_data_payload = st.session_state.dwh_result.get('data')

                            # 5. Generate Final Answer
                            response = generate_final_answer(
                                system_instructions=pkg.system_instructions,
                                customer_data=customer_data_payload,
                                retrieved_docs=retrieved_docs, 
                                user_query=prompt_to_process,
                                chat_history=chat_history_text
                            )
                        
                        st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
