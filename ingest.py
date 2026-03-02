# ΠΑΙΡΝΕΙ ΤΑ PDF ΚΑΙ TXT ΑΠΟ ΤΟ knowledge_base/ folder
# ΤΑ ΧΩΡΙΖΕΙ ΣΕ ΚΟΜΜΑΤΙΑ ΚΑΙ ΤΑ ΒΑΖΕΙ ΣΤΟ VECTOR DB

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load your GEMINI_API_KEY
load_dotenv()

# Define your paths based on your folder structure
SOURCE_DOCS_DIR = "./knowledge_base"
CHROMA_DB_DIR = "./knowledge_base/chroma_db"

def build_vector_database():
    print(f"📁 Looking for documents in {SOURCE_DOCS_DIR}...")
    
    # 1. Load all PDFs and Text files from the knowledge_base folder
    # Note: If you only have text files, you can just use TextLoader.
    pdf_loader = DirectoryLoader(SOURCE_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(SOURCE_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    
    documents = []
    documents.extend(pdf_loader.load())
    documents.extend(txt_loader.load())
    
    if not documents:
        print("⚠️ No documents found! Put some PDFs or TXTs in the knowledge_base folder.")
        return

    print(f"📄 Found {len(documents)} document(s).")

    # 2. Split the documents into smaller semantic chunks
    # 500 characters with a 50-character overlap keeps sentences intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split documents into {len(chunks)} chunks.")

    # 3. Embed the chunks using Gemini and save to ChromaDB
    print("🧠 Embedding chunks and saving to ChromaDB... (This takes a moment)")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # This creates the vector database and saves it to the disk
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"✅ Success! Vector database saved to {CHROMA_DB_DIR}")

if __name__ == "__main__":
    build_vector_database()