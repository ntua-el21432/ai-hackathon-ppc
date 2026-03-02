import pandas as pd
from src.extraction import extract_bill_data
from src.dwh_matcher import match_customer
from src.rag_engine import generate_prompt_package
from src.generator import generate_final_answer
from src.vector_store import retrieve_knowledge

def run_pipeline(bill_text: str, user_query: str, mock_dwh: pd.DataFrame):
    print("--- STEP 1: Extracting Bill Data ---")
    extracted_data = extract_bill_data(bill_text)
    print(f"Extracted: Customer {extracted_data.customer_id}, Amount: €{extracted_data.total_amount}")

    print("\n--- STEP 2: DWH Matching ---")
    dwh_result = match_customer(extracted_data.customer_id, mock_dwh)
    print(f"Match Status: {dwh_result['status']}")

    print("\n--- STEP 3: Generating Prompt Package ---")
    prompt_package = generate_prompt_package(
        user_query=user_query,
        dwh_status=dwh_result['status'],
        extracted_data=extracted_data.model_dump_json()
    )
    print(f"Confidence Score: {prompt_package.confidence_score}")
    print(f"Generated Queries: {prompt_package.retrieval_queries}")

    print("\n--- STEP 3.5: Querying ChromaDB ---")
    if prompt_package.retrieval_queries:
        real_retrieved_docs = retrieve_knowledge(prompt_package.retrieval_queries)
        print(f"Retrieved {len(real_retrieved_docs.split('---'))} chunks of knowledge.")
    else:
        real_retrieved_docs = "No retrieval queries generated."

    print("\n--- STEP 4: Final Grounded Answer ---")
    if dwh_result['status'] == "single_match":
        final_answer = generate_final_answer(
            system_instructions=prompt_package.system_instructions,
            customer_data=dwh_result['data'],
            retrieved_docs=real_retrieved_docs,
            user_query=user_query
        )
        print("\nAGENT RESPONSE:\n", final_answer)
    else:
        print("\nAGENT RESPONSE:\nI need clarification: ", prompt_package.clarifying_questions[0])

if __name__ == "__main__":
    mock_bill = "PPC Invoice. ID: 12345. Period: March. Total: 150.00 EUR. Tariff: MyHome. kWh: 1000."
    mock_db = pd.DataFrame({
        "customer_id": ["12345"], 
        "name": ["Maria K."], 
        "last_month_bill": [120.00]
    })
    user_q = "Why is my bill higher than last month?"
    
    run_pipeline(mock_bill, user_q, mock_db)