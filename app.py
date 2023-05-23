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
     # Extract the question from the request body
    question = request.json.get('question', 'Who is the main character?') # Default question if not provided
    ctxtsearch = contextAccess
    websearch = WebAccess
    prompt_template = """### Instruction:
    You are a librarian AI who uses document information to answer questions. Documents as formatted as follows: [(Document(page_content="<important context>", metadata='source': '<source>'), <rating>)] where <important context> is the context, <source> is the source, and <rating> is the rating. 
    
    Context Search: A way to explore your documents. Useful for when you need to answer questions about your database. The input is the question to search relavant information.

    Strictly use the following format:

    Question: the input question you must answer
    Thought: you should think about the best way to query your documents with Context Search
    Action: the action to take, should be one of [Context Search]
    Action Input: the input to the action, should be a question.
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question


    For examples:
    Question: How old is CEO of Microsoft wife?
    Thought: I need to look in my database to find that
    Action: Context Search
    Action Input: Who is the CEO of Microsoft?
    Observation: Satya Nadella is the CEO of Microsoft.
    Thought: Now, I should find out Satya Nadella's wife.
    Action: Context Search
    Action Input: Who is Satya Nadella's wife?
    Thought: Then, I need to check Anupama Nadella's age.
    Action: Context Search
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
    valid_tools = ['Context Search']

    print("TESTING ONE")
    #context = ctxtsearch.searchContext(question)
    #return context
    # set the default language model used to execute guidance programs
    guidance.llm = guidance.llms.TextGenerationWebUI()
    guidance.llm.caching = False
    prompt = guidance(prompt_template)
    google = websearch.searchGoogle(question)
    result = prompt(question=question, search=ctxtsearch.searchContext, valid_answers=valid_answers, valid_tools=valid_tools)
  
    return str(result)

    # You can return the result as JSON
    return str({'result': executed_program})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
