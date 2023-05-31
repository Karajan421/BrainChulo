from importlib import reload
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
#from constants import *
import guidance
import torch
from model_singleton import load_model_into_guidance
import guidance_agent.tools
from guidance_agent.tools import load_tools
from guidance_agent.agent import CustomAgentGuidance
from pydantic import BaseModel
from typing import Optional
import os
#from langchain.llms import OpenAI
from colorama import Fore, Style
#import uvicorn
import nest_asyncio
from settings import logger, load_config
import asyncio
from dotenv import load_dotenv

load_dotenv()

class Question(BaseModel):
    question: Optional[str] = 'Who is the main character?'

app = FastAPI()
nest_asyncio.apply()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

TEST_FILE = os.getenv("TEST_FILE")
MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE = torch.device('cuda:0')

llama = None
dict_tools = None

@app.post('/load_model')
async def load_model():
    print(Fore.GREEN + Style.BRIGHT + f"Loading model...." + Style.RESET_ALL)
    global llama
    llama = guidance.llms.Transformers(MODEL_PATH, device_map="auto", load_in_8bit=True)
    guidance.llm = llama
    return 'Model loaded successfully'

@app.post('/load_tools')
async def load_tools_route():
    config = load_config()
    global dict_tools
    # Reload the tools module to get the latest version
    reload(guidance_agent.tools)
    print(Fore.GREEN + Style.BRIGHT + f"Loading tools...." + Style.RESET_ALL)
    if llama is None:
        return {'message': 'Model is not loaded. Load the model first', 'status_code': 400}
    dict_tools = load_tools(llama, config, filepath=TEST_FILE)
    return 'Tools loaded successfully' 

@app.post('/run_script')
async def run_script(question: Question = Body(...)):
    global dict_tools
    if dict_tools is None:
        return {'message': 'Tools are not loaded. Load the tools first', 'status_code': 400}
    custom_agent = CustomAgentGuidance(guidance, dict_tools)
    final_answer = custom_agent(question.question)
    if isinstance(final_answer, dict):
        return {'answer': str(final_answer), 'function': str(final_answer['fn'])}
    else:
        # Handle the case when final_answer is not a dictionary.
        return {'answer': str(final_answer)}


@app.post('/reload_modules')
async def reload_modules():
    global guidance_agent
    # Reload the modules
    guidance_agent.tools = reload(guidance_agent.tools)
    guidance_agent.agent = reload(guidance_agent.agent)

    # Re-import functions or classes
    from guidance_agent.tools import load_tools
    from guidance_agent.agent import CustomAgentGuidance
    print(Fore.GREEN + Style.BRIGHT + f"Modules reloaded successfully" + Style.RESET_ALL)
    return 'Modules reloaded successfully'

if __name__ == "__main__":
    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    config = Config()
    config.bind = ["0.0.0.0:5001"]
    asyncio.run(serve(app, config))