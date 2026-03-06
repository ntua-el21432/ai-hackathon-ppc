# NOTES
# 1. Change to being a single function like extraction.py to import directly to main.py
# 2. Use config.py to load the model
# 3. Figure out if this approach or llm.with_structured_output is better for the json output guarantee
# probably: structured_llm = raw_llm.with_structured_output(BillExtraction)
# 4. Encode image is already in helpers.py. No need for it here
# 5. Keep both greek and engligh descriptions for model, or better have a English/ Greek option.

import os
import base64
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- STEP 1: Define the Strict Data Schema ---
class BillLineItem(BaseModel):
    charge_type: str = Field(description="energy, regulated, taxes, or other")
    description: str = Field(description="The exact text description of the charge from the bill")
    quantity_kwh: Optional[float] = None
    amount: float

class ExtractedBill(BaseModel):
    # Guardrails for redacted IDs and ignoring RF codes
    customer_id: Optional[str] = Field(description="Look for 'Κωδικός Εταίρου' or 'Κωδικός Πελάτη'. If it is redacted as 'XXXXXXXXXX', return 'XXXXXXXXXX'. DO NOT return the RF payment code.", default=None)
    contract_id: Optional[str] = Field(description="Look for 'Λογαριασμός Συμβολαίου' or 'Αρ. Παροχής'. If redacted, return 'XXXXXXXXXX'.", default=None)
    tariff_code: Optional[str] = Field(description="The name of the plan, e.g., 'myHome GasControl'", default=None)
    total_amount: float
    line_items: List[BillLineItem]
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the extraction accuracy.")

# --- STEP 2: The Core Vision Agent ---
class EnergyBillAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    def encode_image(self, path):
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode('utf-8')

    def run_extraction(self, image_path: str) -> ExtractedBill:
        print(f" Phase 1: Extracting raw data from image...")
        img_b64 = self.encode_image(image_path)
        
        # Determine MIME type based on extension
        ext = image_path.split('.')[-1].lower()
        mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg'] else "image/jpeg"

        response = self.client.beta.chat.completions.parse(
            model=self.deployment,
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a strict PPC (ΔΕΗ) Billing Data Extractor. "
                        "Your ONLY job is to extract the data accurately from the bill image into JSON format. "
                        "CRITICAL VISION RULES FOR LINE ITEMS: "
                        "- Read the summary table row by row. DO NOT merge rows or shift amounts to make the math work. "
                        "- Look for these EXACT Greek terms and do not misspell them: 'Χρεώσεις Προμήθειας ΔΕΗ', 'Ρυθμιζόμενες Χρεώσεις', 'Έναντι Κατανάλωσης', 'Διάφορα (Φόροι-Τέλη κλπ.)', 'Φ.Π.Α.', 'Προηγούμενο Ανεξόφλητο Ποσό'. "
                        "- Ensure the extracted amount perfectly aligns with its specific row on the bill. For example, 'Φ.Π.Α.' and 'Διάφορα' are separate rows."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all billing details from this image:"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}}
                    ]
                }
            ],
            response_format=ExtractedBill,
            temperature=0.0, # Keep extraction highly deterministic
        )
        return response.choices[0].message.parsed

# --- STEP 3: Run & Export ---
if __name__ == "__main__":
    agent = EnergyBillAgent()
    try:
        final_bill = agent.run_extraction("sample_bill.jpg") 
        
        print("\n" + "="*50)
        print(f"CUSTOMER ID: {final_bill.customer_id}")
        print(f"CONTRACT ID: {final_bill.contract_id}")
        print(f"TARIFF: {final_bill.tariff_code}")
        print(f"TOTAL AMOUNT: €{final_bill.total_amount}")
        print(f"LINE ITEMS FOUND: {len(final_bill.line_items)}")
        print("="*50)

        # Output to JSON (UTF-8 to preserve Greek characters)
        with open("hackathon_output.json", "w", encoding="utf-8") as f:
            f.write(final_bill.model_dump_json(indent=2))
            
        print("\n✅ Successfully saved to hackathon_output.json ready for DB matching!")

    except Exception as e:
        print(f"❌ Pipeline Error: {e}")