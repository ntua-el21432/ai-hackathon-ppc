# ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ΓΙΑ ΝΑ ΧΡΗΣΙΜΟΠΟΗΣΟΥΜΕ ΣΤΟ APP.PY

import base64
import pandas as pd
import os

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Converts raw image bytes (from Streamlit's file_uploader) 
    into a Base64 string required by Vision LLMs.
    """
    return base64.b64encode(image_bytes).decode('utf-8')

def load_mock_dwh(filepath: str = "data/mock_dwh.csv") -> pd.DataFrame:
    """
    Loads the mock Data Warehouse CSV into a Pandas DataFrame.
    If the file doesn't exist yet, it returns a hardcoded fallback dataframe 
    so your app doesn't crash during testing.
    """
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        # Fallback for immediate testing
        print(f"⚠️ Warning: {filepath} not found. Using fallback mock data.")
        return pd.DataFrame({
            "customer_id": ["GR-99824", "GR-11111"], 
            "name": ["Maria K.", "Nikos P."],
            "active_tariff": ["MyHome Online", "MyHome Basic"],
            "avg_kwh_6m": [320, 450],
            "last_3_bills_total": [310.50, 420.00]
        })

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