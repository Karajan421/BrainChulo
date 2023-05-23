import guidance
from collections import namedtuple
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from app.conversations.document_based import DocumentBasedConversation
import nest_asyncio
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from app.llms.oobabooga_llm import OobaboogaLLM
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from app.prompt_templates.document_based_conversation import ConversationWithDocumentTemplate 
from langchain.document_loaders import TextLoader
from app.tools.web_access import WebAccess
from app.tools.context_access import contextAccess


app = Flask(__name__)
CORS(app)

# Apply the nest_asyncio patch at the start of your script.
nest_asyncio.apply()

@app.route('/run_script', methods=['POST'])
@cross_origin()


def run_script():
    ctxtsearch = contextAccess
    websearch = WebAccess
    prompt_template = """### Instruction:
    You are a librarian AI who uses document information to answer questions. Documents as formatted as follows: [(Document(page_content="<important context>", metadata='source': '<source>'), <rating>)] where <important context> is the context, <source> is the source, and <rating> is the rating. 
    Strictly use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: what you should do to answer the question, should a search in Context
    Action Input: the input to the action, should be a question.
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question


    For examples:
    Question: How old is CEO of Microsoft wife?
    Thought: First, I need to find who is the CEO of Microsoft.
    Action: Searching through  [(Document(page_content='Satya Nadella is the CEO of Microsoft Corporation. He took over as CEO in February 2014, succeeding Steve Ballmer.'), 0.95), (Document(page_content='Microsoft, the Redmond-based tech giant, is led by CEO Satya Nadella, who assumed the role in 2014.', 0.91), (Document(page_content='The chief executive officer of Microsoft Corporation is Satya Nadella. Nadella took the helm in 2014 after Steve Ballmer stepped down.'), 0.93)]
    Action Input: Who is the CEO of Microsoft?
    Observation: Satya Nadella is the CEO of Microsoft.
    Thought: Now, I should find out Satya Nadella's wife.
    Action: Searching through [(Document(page_content='Satya Nadella, the CEO of Microsoft, is married to Anu Nadella. They have been married since 1992.'), 0.96),(Document(page_content='Anu Nadella is the wife of Microsoft CEO, Satya Nadella. They have been together for many years.'), 0.94),(Document(page_content='The wife of Satya Nadella, chief executive officer of Microsoft Corporation, is Anu Nadella. They tied the knot in 1992.'), 0.95)]
    Observation: Satya Nadella's wife's name is Anupama Nadella.
    Action Input: Who is Satya Nadella's wife?
    Thought: Then, I need to check Anupama Nadella's age.
    Action: Searching through [(Document(page_content='Anu Nadella, wife of Microsoft CEO Satya Nadella, is 38 years old.'), 0.96),(Document(page_content='38-year-old Anu Nadella is married to Satya Nadella, the CEO of Microsoft.'), 0.94), (Document(page_content='Anu Nadella, spouse of Satya Nadella, Microsoft Corporation's CEO, is currently 38 years old.'), 0.95)][(Document(page_content='Anu Nadella, wife of Microsoft CEO Satya Nadella, is 38 years old.'), 0.96),(Document(page_content='38-year-old Anu Nadella is married to Satya Nadella, the CEO of Microsoft.'), 0.94), (Document(page_content='Anu Nadella, spouse of Satya Nadella, Microsoft Corporation's CEO, is currently 38 years old.'), 0.95)]
    Action Input: How old is Anupama Nadella?
    Thought: I now know the final answer.
    Final Answer: Anupama Nadella is 38 years old.
    ### Input:
    {{question}}

    ### Response:
    Question: {{question}}
    Thought: {{gen 'thought' stop='\\n'}}
    Action: {{select 'tool_name' options=valid_tools}}
    Action Input: {{gen 'actInput' stop='\\n'}}
    Observation:{{search actInput}}
    Thought: {{gen 'thought2' stop='\\n'}}
    Final Answer: {{gen 'final' stop='\\n'}}"""
    print("TESTING ZERO")


    valid_answers = ['Action', 'Final Answer']
    valid_tools = ['Google Search']

  

    
    convo = DocumentBasedConversation()
    print("TESTING ONE")
    
        
        # set the default language model used to execute guidance programs
    guidance.llm = guidance.llms.TextGenerationWebUI()
    guidance.llm.caching = False
    prompt = guidance(prompt_template)
    question='Is Eminem a football player?'
    google = websearch.searchGoogle(question)
    result = prompt(question='Is Eminem a football player?', search=websearch.searchGoogle, valid_answers=valid_answers, valid_tools=valid_tools)
  
    return str(result)

    print("TESTING")
    question = "Whos is Macbeth?"
    file_path = "/home/karajan/Documents/macbeth.txt"
    convo.load_document(file_path)
    context = convo.context_predict(question)

    valid_answers = ['Action', 'Final Answer']
    valid_tools = ['Google Search']

    prompt = guidance(prompt_template)
    result = prompt(question='Is Eminem a football player?', search=websearch.searchGoogle, valid_answers=valid_answers, valid_tools=valid_tools)
    return result
    
    """"
    # Load text file
    with open(file_path, 'r') as file:
        text = file.read()
    title = os.path.basename(file_path)

    docs = {title: text}  
    documents = [Document(page_content=docs[title]) for title in docs]
    # Create new vector store and index it
    vectordb = None
    
    # Split by section, then split by token limmit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    text_splitter = TokenTextSplitter(chunk_size=1000,chunk_overlap=10, encoding_name="cl100k_base")  # may be inexact
    texts = text_splitter.split_documents(texts)

    vectordb = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
    
    
    search = vectordb.similarity_search_with_score(
    question, top_k_docs_for_context=20
        )
    """
    program = guidance(prompt_template)


    

    executed_program = program(
    question=question,
    context=context,   
    async_mode=False
)
    # You can return the result as JSON
    return str({'result': executed_program})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
