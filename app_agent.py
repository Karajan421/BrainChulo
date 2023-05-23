import gradio as gr
import guidance
import torch
from server.model import load_model_main
from server.tools import load_tools
from server.agent import CustomAgentGuidance
import os
from langchain.llms import OpenAI



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

def greet(name):
    final_answer = custom_agent(name)
    return final_answer, final_answer['fn']

model, tokenizer = load_model_main(MODEL_PATH, CHECKPOINT_PATH, DEVICE)
llama = guidance.llms.Transformers(model=model, tokenizer=tokenizer, device=0)
guidance.llm = llama

dict_tools = load_tools(llama)

custom_agent = CustomAgentGuidance(guidance, dict_tools)

# Call the main function from the ingest script

list_outputs = [gr.Textbox(lines=5, label="Reasoning"), gr.Textbox(label="Final Answer")]
demo = gr.Interface(fn=greet, inputs=gr.Textbox(lines=1, label="Input Text", placeholder="Enter a question here..."), 
                    outputs=list_outputs,
                    title="Demo ReAct agent with Guidance",
                    description="The source code can be found at: https://github.com/QuangBK/localLLM_guidance/",
                   examples=examples)
demo.launch(server_name="0.0.0.0", server_port=7863)

CHROMA_SETTINGS = {}  # Fill with your settings
