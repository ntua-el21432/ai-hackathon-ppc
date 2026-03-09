import pytesseract
import base64
import io 
from PIL import Image
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm

class LineItem(BaseModel):
    description: str = Field(description="Description of the charge")
    amount: float = Field(description="The cost of this line item in Euros")
    charge_type: str = Field(description="Categorize strictly as 'energy', 'regulated', 'taxes', or 'other'")

class BillExtraction(BaseModel):
    customer_id: str = Field(description="Unique customer identifier (preserve leading zeros and dashes)")
    contract_account_num: str = Field(description="The number of the contract or the account (preserve leading zeros)")
    billing_period: str = Field(description="The period the bill covers (e.g., 01/01/2026 - 31/01/2026)")
    total_amount: float = Field(description="Total amount due in Euros")
    tariff_code: str = Field(description="The active tariff or plan name")
    consumption_kwh: float = Field(description="Total energy consumed in kWh")
    line_items: List[LineItem] = Field(description="List of every individual charge on the bill")
    bill_summary: str = Field(description="A brief 2-sentence summary of the bill's contents")
    extraction_confidence: float = Field(description="A score between 0.0 and 1.0 representing how confident you are in the extraction (with 1.0 being certainty)")

def extract_bill_data(image_uri: str) -> BillExtraction:
    try:
        if image_uri.startswith("data:image"):
            # Split off the "data:image/jpeg;base64," prefix
            base64_data = image_uri.split(",")[1]
            # Decode the string back into raw bytes
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(img_bytes))
        else:
            img = Image.open(image_uri)
            
        raw_ocr_text = pytesseract.image_to_string(img, lang="ell+eng")
        
    except Exception as e:
        print(f"Warning: OCR failed ({e}). Proceeding with vision only.")
        raw_ocr_text = "OCR failed or not available."

    llm = get_llm()
    structured_llm = llm.with_structured_output(BillExtraction)

    # 3. Create prompt combining both Vision and OCR text
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI Billing Agent for PPC. 
        Analyze the provided electricity bill image and the raw OCR text to extract the exact fields requested.

        CRITICAL RULES:
        1. Categorize all line items strictly into one of these four categories: 'energy', 'regulated', 'taxes', 'other'.
        2. If an important field is illegible or missing, use a logical default (e.g., 'Unknown' or 0.0), but drastically lower your 'extraction_confidence' score.
        3. Provide a clear, structured 'bill_summary' in exactly 2 sentences.
        4. Preserve all leading zeros and dashes in IDs.
        5. Use the raw OCR text to double-check serial numbers, account IDs, and tiny text, but rely on the image for spatial layout and table structure."""),
        

        # We provide a mock OCR input and the exact JSON we expect out. 
        ("human", """Raw OCR Text Helper:
        ---
        ΔΕΗ Α.Ε.
        ΠΕΛΑΤΗΣ: 0001234569
        ΑΡΙΘΜΟΣ ΠΑΡΟΧΗΣ: 0987654321
        ΠΕΡΙΟΔΟΣ ΚΑΤΑΝΑΛΩΣΗΣ: 01/01/2026 - 31/01/2026
        Γ1 Οικιακό Τιμολόγιο
        Κατανάλωση: 350 kWh
        Χρέωση Προμήθειας: 100.00 EUR
        Δίκτυο Διανομής: 30.00 EUR
        ΦΠΑ: 20.50 EUR
        ΤΕΛΙΚΟ ΠΟΣΟ: 150.50
        ---
        
        Extract the required fields from this bill image carefully:"""),
        
        ("assistant", """{{
          "customer_id": "0001234569",
          "contract_account_num": "0987654321",
          "billing_period": "01/01/2026 - 31/01/2026",
          "total_amount": 150.50,
          "tariff_code": "Γ1 Οικιακό Τιμολόγιο",
          "consumption_kwh": 350.0,
          "line_items": [
            {{
              "description": "Χρέωση Προμήθειας",
              "amount": 100.0,
              "charge_type": "energy"
            }},
            {{
              "description": "Δίκτυο Διανομής",
              "amount": 30.0,
              "charge_type": "regulated"
            }},
            {{
              "description": "ΦΠΑ",
              "amount": 20.5,
              "charge_type": "taxes"
            }}
          ],
          "bill_summary": "This bill covers the period of January 2026 under the Γ1 Οικιακό Τιμολόγιο plan. The customer consumed 350 kWh, resulting in a total amount due of 150.50 Euros.",
          "extraction_confidence": 0.98
        }}"""),
       
       ("human", """Raw OCR Text Helper:
        ---
        ΕΚΚΑΘΑΡΙΣΤΙΚΟΣ ΛΟΓΑΡΙΑΣΜΟΣ ΗΛΕΚΤΡΙΚΗΣ ΕΝΕΡΓΕΙΑΣ
        CUSTOMER DETAILS
        Κωδικός Πελάτη: CUST-0001
        Κωδικός Λογαριασμού: ACC-434626
        BILL SUMMARY
        Τιμολόγιο: ΔΕΗ myHome 4All
        Περίοδος Χρέωσης: 19/01/2026 to 15/02/2026
        ΣΥΝΟΛΙΚΟ ΠΟΣΟ ΠΛΗΡΩΜΗΣ: 100,55 €
        CHARGE BREAKDOWN
        1. Energy Supply Charge: € 80.04
        2. Energy Support Charge: € 40.04
        3. Energy Innovation Charge: € 10.04
        4. Energy Basic Charge: € 10.04
        6. Financial Balance Fee: € 00.03
        7. Power Department (Factor): € 00.00
        8. Trade Fees: € 00.00
        YOUR CONSUMPTION: 285 kWh
        ---
        Extract the required fields from this bill image carefully:"""),
        
        ("assistant", """{{
          "customer_id": "CUST-0001",
          "contract_account_num": "ACC-434626",
          "billing_period": "19/01/2026 to 15/02/2026",
          "total_amount": 100.55,
          "tariff_code": "ΔΕΗ myHome 4All",
          "consumption_kwh": 285.0,
          "line_items": [
            {{ "description": "1. Energy Supply Charge", "amount": 80.04, "charge_type": "energy" }},
            {{ "description": "2. Energy Support Charge", "amount": 40.04, "charge_type": "regulated" }},
            {{ "description": "3. Energy Innovation Charge", "amount": 10.04, "charge_type": "other" }},
            {{ "description": "4. Energy Basic Charge", "amount": 10.04, "charge_type": "energy" }},
            {{ "description": "6. Financial Balance Fee", "amount": 0.03, "charge_type": "other" }},
            {{ "description": "7. Power Department (Factor)", "amount": 0.0, "charge_type": "other" }},
            {{ "description": "8. Trade Fees", "amount": 0.0, "charge_type": "other" }}
          ],
          "bill_summary": "This bill for customer CUST-0001 covers the early 2026 period under the ΔΕΗ myHome 4All tariff. A total consumption of 285 kWh was recorded, leading to a total payment amount of 100.55 Euros.",
          "extraction_confidence": 0.99
        }}"""),

        # --- ACTUAL USER INPUT ---
        ("human", [
            {"type": "text", "text": "Raw OCR Text Helper:\n---\n{ocr_text}\n---\n\nExtract the required fields from this bill image carefully:"},
            {"type": "image_url", "image_url": {"url": "{image_uri}"}} 
        ])
    ])
    
    # αυτο το chain παιρνει το prompt, το περναει στο llm που επιλεξαμε 
    # με την get_llm απο το config.py και μετα το δινει στο parser
    # για να επιστρεψει ενα BillExtraction object με τα πεδια που θελουμε
    chain = prompt | structured_llm 

    # 4. Invoke with both variables
    return chain.invoke({
        "image_uri": image_uri,
        "ocr_text": raw_ocr_text
    })