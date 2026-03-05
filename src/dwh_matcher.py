# ΒΡΙΣΚΕΙ ΤΟΥΣ ΠΕΛΑΤΕΣ ΑΠΟ ΤΗΝ ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ ΠΕΛΑΤΩΝ ΠΟΥ ΘΑ ΕΧΟΥΜΕ ΜΕ ΤΗΝ ΛΟΓΙΚΗ ΠΟΥ ΖΗΤΑΕΙ
import pandas as pd

def match_customer(extracted_customer_id: str, mock_dwh: dict)-> dict:
    """
    Searches the DWH for the customer and joins their context, headers, and lines.
    Returns the exact status needed.
    """
    if mock_dwh is None:
        return {"status": "error", "data": None}

    df_context = mock_dwh["context"]
    df_header = mock_dwh["header"]
    df_lines = mock_dwh["lines"]

    if df_context is None or df_header is None or df_lines is None:
        return {"status": "error", "data": None}
    
    extracted_customer_id = str(extracted_customer_id).strip()

    # look for all rows matching the extracted ID
    customer_matches = df_context[df_context['customer_id'] == extracted_customer_id]
    # No match
    if customer_matches.empty:
        return {"status": "no_match", "data": None}
    # Multiple matches
    if len(customer_matches) > 1:
        return {"status": "multiple_matches", "data": None}
    #Single match
    customer_profile = customer_matches.iloc[0].to_dict()

    # get billing headers
    headers = df_header[df_header['customer_id'].astype(str).str.strip() == extracted_customer_id]
    # get the detailed Line Items for those specific bills
    bill_ids = headers['bill_id'].tolist()
    lines = df_lines[df_lines['bill_id'].isin(bill_ids)]

    # This dictionary gets dumped to JSON and fed directly into the Gemini Prompt
    full_customer_data = {
        "customer_profile": customer_profile,
        "billing_history": headers.to_dict(orient="records"),
        "line_items_history": lines.to_dict(orient="records")
    }

    return {"status": "single_match", "data": full_customer_data}