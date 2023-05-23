import gradio as gr
import guidance
import torch
from server.model import load_model_main
from server.tools import load_tools
from server.agent import CustomAgentGuidance
import os
from langchain.llms import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import nest_asyncio

app = Flask(__name__)
CORS(app)


# Apply the nest_asyncio patch at the start of your script.
nest_asyncio.apply()
os.environ["SERPER_API_KEY"] = 'fbac5061b434c6b0e5f55968258b144209993ab2'
MODEL_PATH = '/home/karajan/labzone/textgen/text-generation-webui/models/anon8231489123_vicuna-13b-GPTQ-4bit-128g'
CHECKPOINT_PATH = '/home/karajan/labzone/textgen/text-generation-webui/models/anon8231489123_vicuna-13b-GPTQ-4bit-128g/vicuna-13b-4bit-128g.safetensors'
DEVICE = torch.device('cuda:0')


examples = [
    ["How much is the salary of number 8 of Manchester United?"],
    ["What is the population of Congo?"],
    ["Where was the first president of South Korean born?"],
    ["What is the population of the country that won World Cup 2022?"]    
]

@app.route('/run_script', methods=['POST'])
@cross_origin()


def run_script():
    question = request.json.get('question', 'Who is the main character?') # Default question if not provided


    #model, tokenizer = load_model_main(MODEL_PATH, CHECKPOINT_PATH, DEVICE)
    llama =  guidance.llm = guidance.llms.TextGenerationWebUI()
    guidance.llm.caching = False

    dict_tools = load_tools(llama)

    custom_agent = CustomAgentGuidance(guidance, dict_tools)

    final_answer = custom_agent(question)
    return str(final_answer), str(final_answer['fn'])

CHROMA_SETTINGS = {}  # Fill with your settings

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)