import os
from dotenv import load_dotenv

from app.settings import Settings
from app.tools.base import BaseTool
from langchain.utilities import GoogleSerperAPIWrapper
from app.conversations.document_based import DocumentBasedConversation

# Load .env file
load_dotenv()

class contextAccess():
    def searchContext(t):  
        convo = DocumentBasedConversation()

        # Get file_path from environment variable
        file_path = os.getenv('TEST_FILE')

        convo.load_document(file_path)
        context = convo.context_predict(t)
        return context
