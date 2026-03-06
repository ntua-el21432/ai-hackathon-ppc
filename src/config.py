# LOAD ENVIRONMENTAL VARIABLES FROM .ENV FILE
# ΔΗΛΑΔΗ ΟΤΙ API KEY ΠΟΥ ΘΑ ΧΡΗΣΙΜΟΠΟΙΗΣΟΥΜΕ ΘΑ ΤΟ
# ΕΧΟΥΜΕ ΣΤΟ .ENV ΚΑΙ ΘΑ ΤΟ ΦΟΡΤΩΝΟΥΜΕ ΜΕ ΤΗΝ
# load_dotenv() ΓΙΑ ΝΑ ΜΗΝ ΤΟ ΒΑΖΟΥΜΕ ΜΕΣΑ ΣΤΟΝ ΔΗΜΟΣΙΟ ΚΩΔΙΚΑ
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

def get_gpt_llm():
    """
    Initializes the Azure OpenAI Chat model for LangChain.
    It automatically looks for AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT 
    in your environment.
    """
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), 
        api_version="2024-12-01-preview", 
        temperature=0.0,                 
        max_retries=2                     
    )
    return llm