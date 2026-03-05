# ΦΤΙΑΧΝΕΙ ΤΟ RAG 

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.config import get_llm

class PromptPackage(BaseModel):
    retrieval_queries: List[str] = Field(description="3 distinct search queries to run against the vector database")
    metadata_filters: dict = Field(description="Filters to apply (e.g., {'topic': 'tariffs'})")
    system_instructions: str = Field(description="Strict rules for the final generation (e.g., use citations, avoid hallucination)")
    clarifying_questions: Optional[List[str]] = Field(description="Questions for the user if the DWH match failed or input is ambiguous")
    confidence_score: float = Field(description="Confidence score (0.0 to 1.0) of the overall extraction and matching process")

def generate_prompt_package(user_query: str, dwh_status: str, extracted_data: str, chat_history: str) -> PromptPackage:
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=PromptPackage)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a RAG Strategy Agent. Create a prompt package based on the inputs below.\n{format_instructions}"),
        ("human", "Previous Conversation History: {chat_history}\nUser Query: {user_query}\nDWH Match Status: {dwh_status}\nExtracted Bill Data: {extracted_data}")
    ])
    
    chain = prompt | llm | parser
    return chain.invoke({
        "user_query": user_query,
        "dwh_status": dwh_status,
        "extracted_data": extracted_data,
        "chat_history": chat_history,
        "format_instructions": parser.get_format_instructions()
    })