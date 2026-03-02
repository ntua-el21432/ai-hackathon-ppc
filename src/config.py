# LOAD ENVIRONMENTAL VARIABLES FROM .ENV FILE
# ΔΗΛΑΔΗ ΟΤΙ API KEY ΠΟΥ ΘΑ ΧΡΗΣΙΜΟΠΟΙΗΣΟΥΜΕ ΘΑ ΤΟ
# ΕΧΟΥΜΕ ΣΤΟ .ENV ΚΑΙ ΘΑ ΤΟ ΦΟΡΤΩΝΟΥΜΕ ΜΕ ΤΗΝ
# load_dotenv() ΓΙΑ ΝΑ ΜΗΝ ΤΟ ΒΑΖΟΥΜΕ ΜΕΣΑ ΣΤΟΝ ΔΗΜΟΣΙΟ ΚΩΔΙΚΑ
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )