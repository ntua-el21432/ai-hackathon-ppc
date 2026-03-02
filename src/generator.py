from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm

def generate_final_answer(system_instructions: str, customer_data: dict, retrieved_docs: str, user_query: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a PPC AI Billing Agent. Follow these instructions strictly:
        {system_instructions}
        
        Context Rules:
        1. Use ONLY the retrieved knowledge documents provided.
        2. Provide inline citations like [Source: Document Name].
        3. Do NOT hallucinate or guess.
        
        Customer Profile: {customer_data}
        Retrieved Knowledge: {retrieved_docs}"""),
        ("human", "{user_query}")
    ])
    
    chain = prompt | llm
    return chain.invoke({
        "system_instructions": system_instructions,
        "customer_data": customer_data,
        "retrieved_docs": retrieved_docs,
        "user_query": user_query
    }).content