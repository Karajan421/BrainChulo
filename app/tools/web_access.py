from app.settings import Settings
from app.tools.base import BaseTool
from langchain.utilities import GoogleSerperAPIWrapper
import os

#config = Settings.load_config()
os.environ["SERPER_API_KEY"] = 'fbac5061b434c6b0e5f55968258b144209993ab2'
search = GoogleSerperAPIWrapper()



class WebAccess():
    def searchGoogle(t):    
        return search.run(t)

