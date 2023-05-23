from app.settings import Settings
from app.tools.base import BaseTool
from langchain.utilities import GoogleSerperAPIWrapper
from app.conversations.document_based import DocumentBasedConversation


class contextAccess():
    def searchContext(t):  
        convo = DocumentBasedConversation()
        file_path = "/home/karajan/Documents/macbeth.txt"
        convo.load_document(file_path)
        context = convo.context_predict(t)
        return context