import os
import re
import time

import requests
from InstructorEmbedding import INSTRUCTOR
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import pipeline
from langchain.chains import ConversationChain
from app.conversations.document_based import DocumentBasedConversation 
from app.benchmark.constants import *
from dotenv import load_dotenv

class benchQA:
    question_check_template = """Given the following pieces of context, determine if the question is able to be answered by the information in the context.
Respond with 'yes' or 'no'.
{context}
Question: {question}
"""
    QUESTION_CHECK_PROMPT = PromptTemplate(
        template=question_check_template, input_variables=["context", "question"]
    )
    def __init__(self, config: dict={}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.prompt = None
        self.test_file = os.getenv("TEST_FILE", "")
    # The following class methods are useful to create global GPU model instances
    # This way we don't need to reload models in an interactive app,
    # and the same model instance can be used across multiple user sessions
    @classmethod
    def create_instructor_xl(cls):
        return HuggingFaceInstructEmbeddings(model_name=EMB_INSTRUCTOR_XL, model_kwargs={"device": "cuda"})

    @classmethod
    def create_flan_t5_xxl(cls, load_in_8bit=False):
        # Local flan-t5-xxl with 8-bit quantization for inference
        # Wrap it in HF pipeline for use with LangChain
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-xxl",
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )

    @classmethod
    def create_flan_t5_xl(cls, load_in_8bit=False):
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-xl",
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )

    @classmethod
    def create_fastchat_t5_xl(cls, load_in_8bit=False):
        return pipeline(
            task="text2text-generation",
            model = "lmsys/fastchat-t5-3b-v1.0",
            model_kwargs={"device_map": "auto", "load_in_8bit": load_in_8bit, "max_length": 512, "temperature": 0.}
        )

    def init_models(self) -> None:
        """ Initialize new models based on config """
        load_in_8bit = self.config["load_in_8bit"]

        if self.config["embedding"] == EMB_OPENAI_ADA:
            # OpenAI ada embeddings API
            self.embedding = OpenAIEmbeddings()
        elif self.config["embedding"] == EMB_INSTRUCTOR_XL:
            # Local INSTRUCTOR-XL embeddings
            if self.embedding is None:
                self.embedding = benchQA.create_instructor_xl()
        else:
            raise ValueError("Invalid config")

        if self.config["llm"] == LLM_OPENAI_GPT35:
            # OpenAI GPT 3.5 API
            pass
        elif self.config["llm"] == LLM_FLAN_T5_XL:
            if self.llm is None:
                self.llm = benchQA.create_flan_t5_xl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FLAN_T5_XXL:
            if self.llm is None:
                self.llm = benchQA.create_flan_t5_xxl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == LLM_FASTCHAT_T5_XL:
            if self.llm is None:
                self.llm = benchQA.create_fastchat_t5_xl(load_in_8bit=load_in_8bit)
        elif self.config["llm"] == OOBA:
            if self.llm is None:
                self.llm = OobaboogaLLM()
        else:
            raise ValueError("Invalid config")

    def search_and_read_page(self, file_path: str) -> tuple[str, str]:
        """
        Searches benchQA for the given query, take the first result
        Then chunks the text of it and indexes it into a vector store

        Returns the title and text of the page
        """
        # Search benchQA and get first result
        file_path = self.test_file
        # Load text file
        with open(file_path, 'r') as file:
            text = file.read()
            
        # Use filename as title
        title = os.path.basename(file_path)
        docs = {title: text}

        # Create new vector store and index it
        self.vectordb = None
        documents = [Document(page_content=docs[title]) for title in docs]

        # Split by section, then split by token limmit
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        text_splitter = TokenTextSplitter(chunk_size=1000,chunk_overlap=10, encoding_name="cl100k_base")  # may be inexact
        texts = text_splitter.split_documents(texts)

        self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding)

        # Create the LangChain chain
        if self.config["llm"] == LLM_OPENAI_GPT35:
            # Use ChatGPT API
            self.qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.), chain_type="stuff",\
                                        retriever=self.vectordb.as_retriever(search_kwargs={"k":4}))
        else:
            # Use local LLM
            if self.config["llm"] != OOBA:
                hf_llm = HuggingFacePipeline(pipeline=self.llm)
            else:
                hf_llm = self.llm
                self.doc_conv = DocumentBasedConversation()
                self.doc_conv.load_document(file_path)
                return title, text
            self.qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",\
                                        retriever=self.vectordb.as_retriever(search_kwargs={"k":4}))
            print(self.qa)
            if self.config["question_check"]:
                self.q_check = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff",\
                             retriever=self.vectordb.as_retriever(search_kwargs={"k":4}))
                self.q_check.combine_documents_chain.llm_chain.prompt = benchQA.QUESTION_CHECK_PROMPT

        title = title
        text = text
        return title, text



    def get_answer(self, question: str) -> str:
        print(f"Question: {question}")
        print(self.config["llm"])
        if self.config["llm"] == OOBA:
           
            # Use the predict method
            answer = self.doc_conv.predict(question)
            print(answer)
            return answer
        elif self.config["llm"] != LLM_OPENAI_GPT35 and self.config["question_check"]:
            # For local LLMs, do a self-check to see if question can be answered
            # If unanswerable, respond with "I don't know"
            answerable = self.q_check.run(question)
            if self.config["llm"] == LLM_FASTCHAT_T5_XL:
                answerable = self._clean_fastchat_t5_output(answerable)
                print(answerable)
            if answerable != "yes":
                return "I don't know"
        
        # Answer the question
        answer = self.qa.run(question)
        if self.config["llm"] == LLM_FASTCHAT_T5_XL:
            answer = self._clean_fastchat_t5_output(answer)
        return answer

    def _clean_fastchat_t5_output(self, answer: str) -> str:
        # Remove <pad> tags, double spaces, trailing newline
        answer = re.sub(r"<pad>\s+", "", answer)
        answer = re.sub(r"  ", " ", answer)
        answer = re.sub(r"\n$", "", answer)
        return answer
