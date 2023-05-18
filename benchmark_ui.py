import os
import re
import time

from InstructorEmbedding import INSTRUCTOR
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores import Chroma
from transformers import pipeline

from app.benchmark.constants import *
from app.benchmark.test_queries import benchQA
from app.llms.oobabooga_llm import OobaboogaLLM

from dotenv import load_dotenv
import gradio as gr

# Load OpenAI API key
load_dotenv()

# Global model instances
instructor_xl = None
flan_t5_xl = None
flan_t5_xxl = None
fastchat_t5_xl = None

with gr.Blocks() as qa_app:
    # Initialization
    qa = gr.State(benchQA({"question_check": True, "load_in_8bit": False}))

    # Layout
    gr.Markdown("""# BrainChulo Benchmark
    Use OpenAI and/or local models for embeddings/LLM
    """)
    
    with gr.Tab("Model setup"):
        gr.Markdown("""Select embedding and LLM models
        """)
        with gr.Row() as row:
            with gr.Column():
                emb_radio = gr.Radio([EMB_OPENAI_ADA, EMB_INSTRUCTOR_XL],
                         label="Select embedding model")
                llm_radio = gr.Radio([LLM_OPENAI_GPT35, LLM_FLAN_T5_XL, LLM_FLAN_T5_XXL, LLM_FASTCHAT_T5_XL, OOBA],
                         label="Select LLM model",
                        info="Note: flan-t5-xxl will run out of memory on a single A10G")
            with gr.Column():
                model_text_box = gr.Textbox(label="Current models")
                model_load_btn = gr.Button("Load models")

    with gr.Tab("Read Database"):
        gr.Markdown("""Search database and print the results""")
        with gr.Row() as row:
            with gr.Column():
                query = gr.Textbox(label="Search query")
                search_btn = gr.Button("Search")
            with gr.Column():
                title_box = gr.Textbox(label="Doc title")
                text_box = gr.Textbox(label="Doc text")
                
    with gr.Tab("Q&A"):
        gr.Markdown("""Ask a question about the database""")
        with gr.Row() as row:
            with gr.Column():
                question = gr.Textbox(label="Enter your question")
                question_btn = gr.Button("Ask")
            with gr.Column():
                answer_box = gr.Textbox(label="Answer")

    # Logic
    # Model setup
    def load_model(emb, llm, qa):
        global instructor_xl
        global flan_t5_xl
        global flan_t5_xxl
        global fastchat_t5_xl
        
        if emb == EMB_OPENAI_ADA:
            qa.embedding = OpenAIEmbeddings()
        elif emb == EMB_INSTRUCTOR_XL:
            if instructor_xl is None:
                instructor_xl = benchQA.create_instructor_xl()
            qa.embedding = instructor_xl
        else:
            raise ValueError("Invalid embedding setting")
        qa.config["embedding"] = emb
            
        if llm == LLM_OPENAI_GPT35:
            pass
        elif llm == LLM_FLAN_T5_XL:
            if flan_t5_xl is None:
                flan_t5_xl = benchQA.create_flan_t5_xl()
            qa.llm = flan_t5_xl
        elif llm == LLM_FLAN_T5_XXL:
            if flan_t5_xxl is None:
                flan_t5_xxl = benchQA.create_flan_t5_xxl()
            qa.llm = flan_t5_xxl
        elif llm == LLM_FASTCHAT_T5_XL:
            if fastchat_t5_xl is None:
                fastchat_t5_xl = benchQA.create_fastchat_t5_xl()
            qa.llm = fastchat_t5_xl
        elif llm == OOBA:
            qa.llm = OobaboogaLLM()
        else:
            raise ValueError("Invalid LLM setting")
        qa.config["llm"] = llm
        
        return f"Embedding model: {emb}\nLLM: {llm}"
        
    model_load_btn.click(
        load_model,
        [emb_radio, llm_radio, qa],
        [model_text_box]
    )
    
    # Search benchQA
    def db_search(query, qa):
        title, text = qa.search_and_read_page(query)
        return {
            title_box: title,
            text_box: text,
        }
    
    search_btn.click(
        db_search, 
        [query, qa],
        [title_box, text_box]
    )
    
    # Q&A
    def qa_fn(question, qa):
        answer = qa.get_answer(question)
        return {answer_box: answer}
    
    question_btn.click(
        qa_fn,
        [question, qa],
        [answer_box]
        
    )
qa_app.launch(debug=True)
