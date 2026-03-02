# EXTRACT INFO FROM THE IMAGE AND PARSE IT WITH PYDANTIC
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import get_llm

class BillExtraction(BaseModel):
    customer_id: str = Field(description="Unique customer identifier")
    billing_period: str = Field(description="The period the bill covers")
    total_amount: float = Field(description="Total amount due in Euros")
    tariff_code: str = Field(description="The active tariff or plan name")
    consumption_kwh: float = Field(description="Total energy consumed in kWh")

def extract_bill_data(image_uri: str) -> BillExtraction:
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=BillExtraction)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert OCR and data extraction agent for PPC utility bills.\n{format_instructions}"),
        ("human", [
            {"type": "text", "text": "Extract the required fields from this bill image:"},
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