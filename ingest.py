import os
from dotenv import load_dotenv
# 🌟 NEW: The modern, warning-free Azure import!
from langchain_community.document_loaders import AzureBlobStorageContainerLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

def build_azure_vector_database():
    print("☁️ Connecting to Azure Blob Storage...")
    
    # 1. Load documents directly from your Azure Blob Container
    loader = AzureBlobStorageContainerLoader(
        conn_str=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        container=os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    )
    
    documents = loader.load()
    
    if not documents:
        print("⚠️ No documents found in the Azure Blob container!")
        return

    print(f"📄 Fetched {len(documents)} document(s) from the cloud.")

    # 2. Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Split documents into {len(chunks)} chunks.")

    # 3. Initialize Azure OpenAI Embeddings
    print("🧠 Initializing Azure OpenAI Embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    
    # 4. Connect to Azure AI Search
    print("🔍 Connecting to Azure AI Search...")
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query,
    )
    
    # 5. Upload everything to the vector database
    print(f"🚀 Uploading {len(chunks)} chunks to Azure AI Search index '{os.getenv('AZURE_SEARCH_INDEX_NAME')}'...")
    vector_store.add_documents(chunks)
            
    print("✅ Success! Knowledge base fully uploaded and indexed in Azure.")

if __name__ == "__main__":
    build_azure_vector_database()