# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ΓΙΑ ΝΑ ΧΡΗΣΙΜΟΠΟΗΣΟΥΜΕ ΣΤΟ APP.PY

import base64
import pandas as pd
import os
import streamlit as st

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Converts raw image bytes (from Streamlit's file_uploader) 
    into a Base64 string required by Vision LLMs.
    """
    return base64.b64encode(image_bytes).decode('utf-8')

#@st.cache_data
def load_mock_dwh():
    """
    Loads the mock Data Warehouse CSV into a Pandas DataFrame.
    """
    try:
        return {
            "context": pd.read_csv("data/Customer_Context.csv"),
            "header": pd.read_csv("data/Billing_Header.csv"),
            "lines": pd.read_csv("data/Billing_Lines.csv")
        }
    except Exception as e:
        st.error(f"Error loading databases: {e}")
        return None

def format_currency(amount: float) -> str:
    """Formats a raw float into a clean Euro string for the UI."""
    return f"€{amount:,.2f}"

def validate_environment():
    """
    A quick sanity check to run when the app starts. 
    It warns you if you forgot to set your API keys.
    """
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
        return False, "⚠️ No API key found. Please check your .env file."
    return True, "API keys loaded."