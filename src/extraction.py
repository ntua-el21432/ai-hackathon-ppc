# EXTRACT INFO FROM THE IMAGE AND PARSE IT WITH PYDANTIC
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import get_llm

class LineItem(BaseModel):
    description: str = Field(description="Description of the charge")
    amount: float = Field(description="The cost of this line item in Euros")
    charge_type: str= Field(description="Categorize strictly as 'energy', 'regulated', 'taxes', or 'other'")

class BillExtraction(BaseModel):
    customer_id: str = Field(description="Unique customer identifier (preserve leading zeros and dashes)")
    contract_account_num: str = Field(description="The number of the contract or the account (preserve leading zeros)")
    billing_period: str = Field(description="The period the bill covers (e.g., 01/01/2026 - 31/01/2026)")
    total_amount: float = Field(description="Total amount due in Euros")
    tariff_code: str = Field(description="The active tariff or plan name")
    consumption_kwh: float = Field(description="Total energy consumed in kWh")
    line_items: List[LineItem] = Field ( description="List of every individual charge on the bill")
    bill_summary: str= Field(description="A brief 2-sentence summary of the bill's contents")
    extraction_confidence: float= Field(description="A score between 0.0 and 1.0 representing how confident you are in the extraction (with 1.0 being certainty)")

def extract_bill_data(image_uri: str) -> BillExtraction:
    llm = get_llm()
    parser= PydanticOutputParser(pydantic_object=BillExtraction)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI Billing Agent for PPC. 
        Analyze the provided electricity bill image and extract the exact fields requested.
        {format_instructions}
        CRITICAL RULES:
        1. Categorize all line items strictly into one of these four categories: 'energy', 'regulated', 'taxes', 'other'.
        2. If an important field is illegible or missing, use a logical default (e.g., 'Unknown' or 0.0), but drastically lower your 'extraction_confidence' score.
        3. Provide a clear, structured 'bill_summary' in exactly 2 sentences.
        4. Preserve all leading zeros and dashes in IDs."""),
        ("human", [
            {"type": "text", "text": "Extract the required fields from this bill image carefully:"},
            {"type": "image_url", "image_url": {"url": "{image_uri}"}} 
        ])
    ])
    # αυτο το chain παιρνει το prompt, το περναει στο llm που επιλεξαμε 
    # με την get_llm απο το config.py και μετα το δινει στο parser
    # για να επιστρεψει ενα BillExtraction object με τα πεδια που θελουμε
    chain = prompt | llm | parser

    return chain.invoke({
        "image_uri": image_uri,
        "format_instructions": parser.get_format_instructions()
    })