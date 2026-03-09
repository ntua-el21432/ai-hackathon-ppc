from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm

def generate_final_answer(system_instructions, customer_data, retrieved_docs, user_query, chat_history):
    """
    Generates a grounded, personalized, and cited response.
    """
    llm = get_llm()

    # We use a structured prompt to ensure the LLM follows the "No speculation" rule.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """{system_instructions}

        CONTEXTUAL DATA:
        ---
        CUSTOMER PROFILE & HISTORY:
        {customer_data}
        
        KNOWLEDGE CORPUS (Grounding):
        {retrieved_docs}
         
        PREVIOUS CONVERSATION HISTORY:
        {chat_history}
        
        FINAL INSTRUCTIONS:
        1. Start by addressing the user's specific question.
        2. Use the PREVIOUS CONVERSATION HISTORY to understand question context.
        3. Use the CUSTOMER PROFILE to personalize the answer (e.g., mention their specific tariff).
        4. If the user asks about costs, break down the Line Items clearly.
        5. Always cite the Knowledge Corpus if explaining a fee (e.g., [FAQ Section 2.1]).
        6. If there is a discrepancy between current and past bills, point it out.
        7. DO NOT HALLUCINATE. If the answer isn't in the context, say you don't know and ask for clarification.
        """),
        ("human", "{user_query}")
    ])

    chain = prompt | llm
    
    response = chain.invoke({
        "system_instructions": system_instructions,
        "customer_data": customer_data,
        "retrieved_docs": retrieved_docs,
        "chat_history": chat_history,
        "user_query": user_query
    })

    return response.content