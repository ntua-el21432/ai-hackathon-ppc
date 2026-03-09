from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

from utils.helpers import encode_image_to_base64, load_mock_dwh
from src.extraction import extract_bill_data
from src.dwh_matcher import match_customer
from src.rag_engine import generate_prompt_package
from src.vector_store import retrieve_knowledge
from src.generator import generate_final_answer

app = FastAPI(title="PPC Billing AI Engine")

class ChatRequest(BaseModel):
    prompt: str
    chat_history: str
    dwh_status: str
    extracted_data_json: str

@app.post("/api/v1/extract")
async def extract_and_match(file: UploadFile = File(...)):
    """Receives an image, runs OCR, and matches with DWH."""
    contents = await file.read()
    base64_image = encode_image_to_base64(contents)
    image_uri = f"data:image/jpeg;base64,{base64_image}"
    
    extracted_data = extract_bill_data(image_uri)
    mock_dwh = load_mock_dwh()
    dwh_result = match_customer(extracted_data.customer_id, mock_dwh)
    return {
        "extracted_data": extracted_data.model_dump(),
        "dwh_result": dwh_result
    }

@app.post("/api/v1/chat")
async def chat_with_agent(req: ChatRequest):
    """Handles the RAG routing and final answer generation."""
    pkg = generate_prompt_package(
        user_query=req.prompt,
        dwh_status=req.dwh_status,
        extracted_data=req.extracted_data_json,
        chat_history=req.chat_history
    )
    
    retrieved_docs = retrieve_knowledge(pkg.retrieval_queries)
    
    if pkg.clarifying_questions and len(pkg.clarifying_questions) > 0:
        return {"response": f"**I need clarification:** {pkg.clarifying_questions[0]}"}
    
    response = generate_final_answer(
        system_instructions=pkg.system_instructions,
        customer_data="Data", # Simplified for this example
        retrieved_docs=retrieved_docs,
        user_query=req.prompt,
        chat_history=req.chat_history
    )
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)