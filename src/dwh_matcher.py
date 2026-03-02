# ΒΡΙΣΚΕΙ ΤΟΥΣ ΠΕΛΑΤΕΣ ΑΠΟ ΤΗΝ ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ ΠΕΛΑΤΩΝ ΠΟΥ ΘΑ ΕΧΟΥΜΕ ΜΕ ΤΗΝ ΛΟΓΙΚΗ ΠΟΥ ΖΗΤΑΕΙ
import pandas as pd

def match_customer(extracted_id: str, dwh_df: pd.DataFrame) -> dict:
    matches = dwh_df[dwh_df['customer_id'] == extracted_id]
    # κατι τετοιο θελουμε, αν δειτε ζηταει να εχουμε διαφορετικα αποτελεσματα
    # αναλογα με το αν εχουμε 1, πολλα ή κανενα match
    if len(matches) == 1:
        return {"status": "single_match", "data": matches.to_dict('records')[0]}
    elif len(matches) > 1:
        return {"status": "multiple_matches", "data": None}
    else:
        return {"status": "no_match", "data": None}