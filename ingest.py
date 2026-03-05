# ΠΑΙΡΝΕΙ ΤΑ PDF ΚΑΙ MD ΑΠΟ ΤΟ knowledge_base/ folder
# ΤΑ ΧΩΡΙΖΕΙ ΣΕ ΚΟΜΜΑΤΙΑ ΚΑΙ ΤΑ ΒΑΖΕΙ ΣΤΟ VECTOR DB

import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load your GEMINI_API_KEY
load_dotenv()

# Define your paths based on your folder structure
SOURCE_DOCS_DIR = "./knowledge_base"
CHROMA_DB_DIR = "./knowledge_base/chroma_db"

def build_vector_database():
    print(f"📁 Looking for documents in {SOURCE_DOCS_DIR}...")
    
    # 1. Load all PDFs and Markdown files
    pdf_loader = DirectoryLoader(SOURCE_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    # Force UTF-8 encoding for text/markdown files
    md_loader = DirectoryLoader(
        SOURCE_DOCS_DIR, 
        glob="**/*.md", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    
    documents = []
    documents.extend(pdf_loader.load())
    documents.extend(md_loader.load())
    
    if not documents:
        print("⚠️ No documents found! Put some PDFs or MDs in the knowledge_base folder.")
        return

    print(f"📄 Found {len(documents)} document(s).")

    # 2. Split the documents into smaller semantic chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split documents into {len(chunks)} chunks.")

    # 3. Embed the chunks using Gemini and save to ChromaDB
    print("🧠 Embedding chunks and saving to ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # Initialize an empty Chroma vector store pointing to your directory
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # 🌟 THE BULLETPROOF FIX: 80 chunks per 65 seconds to respect the 100/min quota
    BATCH_SIZE = 80 
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        total_batches = ((len(chunks) - 1) // BATCH_SIZE) + 1
        
        print(f"⏳ Processing batch {current_batch_num} / {total_batches} (Chunks {i} to {i+len(batch)})...")
        
        # Add this batch to the database
        db.add_documents(batch)
        
        # If there are more chunks left, do a hard 65-second sleep to clear the API bucket
        if i + BATCH_SIZE < len(chunks):
            print("💤 API Cooldown: Sleeping for 65 seconds to clear the 60-second rolling quota...")
            time.sleep(65)
            
    print(f"✅ Success! Vector database saved to {CHROMA_DB_DIR}")

if __name__ == "__main__":
    build_vector_database()