import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORT YOUR REAL CORE LOGIC ---
from utils.helpers import encode_image_to_base64, load_mock_dwh, format_currency
from src.extraction import extract_bill_data
from src.dwh_matcher import match_customer
from src.rag_engine import generate_prompt_package
from src.vector_store import retrieve_knowledge
from src.generator import generate_final_answer

# Load environment variables
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="PPC AI Billing Agent", page_icon="⚡", layout="wide")

# --- SESSION STATE INITIALIZATION & RESETS ---
def reset_session():
    """Clears the chat and data when a new bill is uploaded."""
    st.session_state.extracted_data = None
    st.session_state.dwh_result = None
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "dwh_result" not in st.session_state:
    st.session_state.dwh_result = None

# --- SIDEBAR: DEMO CONTROLS ---
with st.sidebar:
    st.header("🛠️ Demo Controls")
    st.markdown("Use this to test the challenge constraints.")
    
    force_dwh = st.radio(
        "Force DWH Match Status:", 
        options=["Auto (Real Lookup)", "single_match", "multiple_matches", "no_match"],
        help="Overrides the Pandas DB lookup to test routing logic."
    )
    
    st.divider()
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ GOOGLE_API_KEY not found in .env file.")

# --- MAIN UI LAYOUT ---
st.title("⚡ PPC AI Billing Assistant")
st.markdown("Upload your electricity bill to get structured insights and ask questions grounded in company policy.")

col_left, col_right = st.columns([1, 1.2], gap="large")

# ==========================================
# LEFT COLUMN: Document Upload & Viewer
# ==========================================
with col_left:
    st.subheader("📄 1. Upload Bill")
    uploaded_file = st.file_uploader(
        "Upload your bill (PNG, JPG)", 
        type=["png", "jpg", "jpeg"],
        on_change=reset_session # Clears chat if a new file is uploaded
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Bill Document", width="stretch")
        
        # Trigger Extraction only once per upload
        if st.session_state.extracted_data is None:
            with st.spinner("Extracting bill data using Vision Model..."):
                try:
                    # 1. Convert image for the LLM
                    base64_image = encode_image_to_base64(uploaded_file.getvalue())
                    
                    # 2. Extract Data (Real LangChain Call)
                    # Note: We prefix with standard data URI format for Gemini multimodal prompts
                    image_uri = f"data:image/jpeg;base64,{base64_image}"
                    st.session_state.extracted_data = extract_bill_data(image_uri)
                    
                    # 3. DWH Lookup (Real Pandas Call)
                    mock_dwh = load_mock_dwh()
                    st.session_state.dwh_result = match_customer(st.session_state.extracted_data.customer_id, mock_dwh)
                    
                    st.success("Data extracted successfully!")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

# ==========================================
# RIGHT COLUMN: Data Summary & Chat
# ==========================================
with col_right:
    if st.session_state.extracted_data:
        data = st.session_state.extracted_data
        
        # --- UI STEP 2: Show Extracted Data ---
        st.subheader("🔍 2. Bill Summary")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Customer ID", data.customer_id)
        m2.metric("Total Amount", format_currency(data.total_amount))
        m3.metric("Period", data.billing_period)
        m4.metric("Tariff", data.tariff_code)
        
        # Handle the Sidebar Override for DWH Status
        dwh_status = st.session_state.dwh_result['status'] if force_dwh == "Auto (Real Lookup)" else force_dwh
        
        if dwh_status == "single_match":
            st.info("✅ **DWH Match:** Customer found. Context loaded securely.", icon="🔐")
        elif dwh_status == "multiple_matches":
            st.warning("⚠️ **DWH Match:** Multiple accounts found for this ID. Clarification needed.", icon="⚠️")
        else:
            st.error("❌ **DWH Match:** No matching customer found in database.", icon="❌")

        # --- UI STEP 3: Chat Interface ---
        st.subheader("💬 3. Ask the AI Agent")
        
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("E.g., Why is this bill higher than last month?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing rules and history..."):
                        
                        # 1. Generate Prompt Package (Real LLM Call)
                        pkg = generate_prompt_package(
                            user_query=prompt,
                            dwh_status=dwh_status,
                            extracted_data=data.model_dump_json()
                        )
                        
                        with st.expander("⚙️ System: View RAG Prompt Package & Confidence"):
                            st.write(f"**Confidence Score:** {pkg.confidence_score * 100}%")
                            st.write("**Generated Retrieval Queries:**", pkg.retrieval_queries)
                            st.write(f"**Routing Status:** {dwh_status}")
                        
                        # 2. Retrieve Knowledge (Real ChromaDB Search)
                        retrieved_docs = retrieve_knowledge(pkg.retrieval_queries)
                        
                        with st.expander("📚 View Retrieved Knowledge (ChromaDB)"):
                            if retrieved_docs:
                                st.markdown(retrieved_docs)
                            else:
                                st.write("No documents retrieved (Database might be empty).")
                        
                        # 3. Generate Final Answer (Real LLM Call)
                        if dwh_status == "single_match" or dwh_status == "no_match":
                            # We still answer on no_match, but strictly using retrieved docs per challenge rules
                            response = generate_final_answer(
                                system_instructions=pkg.system_instructions,
                                customer_data=st.session_state.dwh_result['data'] if dwh_status == "single_match" else "No Data",
                                retrieved_docs=retrieved_docs,
                                user_query=prompt
                            )
                        else:
                            # Multiple matches -> ask clarifying question
                            response = f"**I need clarification:** {pkg.clarifying_questions[0]}"
                        
                        st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("👈 Please upload a bill image on the left to begin the analysis.")