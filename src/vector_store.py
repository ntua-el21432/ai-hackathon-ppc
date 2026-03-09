import os
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langsmith import traceable

@traceable(run_type="retriever", name="Azure Semantic Hybrid Search")
def retrieve_knowledge(queries: list) -> str:
    """Fetches relevant documents from Azure AI Search using Azure OpenAI Embeddings."""
    if not queries:
        return "No retrieval queries generated."

    # 1. Initialize Azure Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    
    # 2. Connect to the Azure AI Search Index
    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function=embeddings.embed_query,
        semantic_configuration_name="semantic" 
    )
    
    retrieved_texts = []

    # 3. Search the database for each query
    for query in queries:
        results = vector_store.semantic_hybrid_search_with_score_and_rerank(query, k=4)
        for doc, search_score, reranker_score in results:
            
            # Since we don't have 'title', we grab the 'source' (file path) and clean it up
            raw_source = doc.metadata.get('source', 'PPC_Document')
            # This splits the path and grabs just the filename at the end
            clean_filename = raw_source.split("\\")[-1].split("/")[-1] 
            
            print(f"📄 Found: {clean_filename} | 🧠 Semantic Score: {reranker_score}")
            
            retrieved_texts.append(doc.page_content)
        # Perform similarity search (fetching top 3 chunks per query)
        #docs = vector_store.similarity_search(query, k=4)
    #     docs = vector_store.semantic_hybrid_search(
    #     query, 
    #     k=4
    # )
        # for doc in docs:
        #     reranker_score = doc.metadata.get('@search.reranker_score', 'N/A')
        #     doc_title = doc.metadata.get('title', 'Unknown Document')
        #     # Print it to your terminal so you can see it live!
        #     print(f"📄 Found: {doc_title} | 🧠 Semantic Score: {reranker_score}")

        #     retrieved_texts.append(doc.page_content)
            
    # 4. Deduplicate the chunks and format them cleanly for the LLM
    unique_texts = list(set(retrieved_texts))
    return "\n\n---\n\n".join(unique_texts)