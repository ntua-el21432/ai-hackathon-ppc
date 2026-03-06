import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

SOURCE_DOCS_DIR = "./knowledge_base"

def build_azure_vector_database():
    print(f"📁 Looking for documents in {SOURCE_DOCS_DIR}...")
    
    # 1. Load all PDFs and Markdown files
    pdf_loader = DirectoryLoader(SOURCE_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
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

    # 2. Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split documents into {len(chunks)} chunks.")

    # 3. Initialize Azure Embeddings & Azure AI Search
    print("☁️ Connecting to Azure AI Search...")
    
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query,
    )
    
    # 4. Upload everything in one shot
    print(f"🚀 Uploading all {len(chunks)} chunks to Azure AI Search...")
    vector_store.add_documents(chunks)
            
    print("✅ Success! Knowledge base uploaded to Azure AI Search.")

if __name__ == "__main__":
    build_azure_vector_database()