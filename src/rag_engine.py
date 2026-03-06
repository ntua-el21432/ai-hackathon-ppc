from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm

class PromptPackage(BaseModel):
    retrieval_queries: List[str] = Field(
        description="1 to 3 distinct search queries to run against the vector database based on the user's latest message and history."
    )
    # 🌟 FIX: Removed the crashing 'metadata_filters' dictionary entirely!
    
    system_instructions: str = Field(
        default="You are a helpful PPC billing assistant. Base your answers strictly on the retrieved documents and customer data. Do not hallucinate.",
        description="Strict rules for the final generation (e.g., use citations, avoid hallucination)."
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Questions for the user if the DWH match failed or input is ambiguous. Leave empty if intent is clear."
    )
    confidence_score: float = Field(
        default=1.0,
        description="Confidence score (0.0 to 1.0) of understanding the user's intent."
    )

def generate_prompt_package(user_query: str, dwh_status: str, extracted_data: str, chat_history: str) -> PromptPackage:
    llm = get_llm()
    
    # 🌟 FIX: Add strict=False to ensure Azure OpenAI doesn't block your schema
    structured_llm = llm.with_structured_output(PromptPackage, strict=False)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert RAG Strategy Agent for a PPC Billing Assistant.
        Your job is to analyze the user's current query, the conversation history, and the extracted bill data to formulate a plan for the final generation agent.
        
        CRITICAL RULES:
        1. Look at the PREVIOUS CONVERSATION HISTORY. If the user's query is short (like "Yes"), use the history to figure out what they mean.
        2. Formulate 'retrieval_queries' that will fetch relevant GENERAL knowledge base articles. 
           🚨 EXTREMELY IMPORTANT: Do NOT put specific customer IDs or exact consumption amounts into the queries. Ask general policy questions like "What are the rules for the MyHome Online tariff?".
        3. You MUST return ALL fields requested in the JSON schema."""),
        
        ("human", """
        PREVIOUS CONVERSATION HISTORY:
        {chat_history}
        
        EXTRACTED BILL DATA: 
        {extracted_data}
        
        DWH MATCH STATUS: 
        {dwh_status}
        
        LATEST USER QUERY: 
        {user_query}
        """)
    ])
    
    # Notice we removed the extra parser here, structured_llm handles it!
    chain = prompt | structured_llm
    
    return chain.invoke({
        "user_query": user_query,
        "dwh_status": dwh_status,
        "extracted_data": extracted_data,
        "chat_history": chat_history
    })