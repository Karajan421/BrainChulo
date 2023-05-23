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

app = Flask(__name__)
CORS(app)

# Apply the nest_asyncio patch at the start of your script.
nest_asyncio.apply()

@app.route('/run_script', methods=['POST'])
@cross_origin()


def run_script():
    print("TESTING ZERO")

    data = request.get_json()
    convo = DocumentBasedConversation()
    print("TESTING ONE")
    question = data.get('question')
    #context = data.get('context')
        
        # set the default language model used to execute guidance programs
    guidance.llm = guidance.llms.TextGenerationWebUI()
    guidance.llm.caching = False
    print("TESTING")

    file_path = "/home/karajan/Documents/olly_tuteur_dataset/french_litterature/horla.txt"
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
    question = "Who's the hero?"
    
    context = vectordb.similarity_search_with_score(
    question, top_k_docs_for_context=20
        )
    
    program = guidance("""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction:
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
    Action: Searching through [Satya Nadella is the CEO of Microsoft Corporation. He took over as CEO in February 2014, succeeding Steve Ballmer, Microsoft, the Redmond-based tech giant, is led by CEO Satya Nadella, who assumed the role in 2014, The chief executive officer of Microsoft Corporation is Satya Nadella. Nadella took the helm in 2014 after Steve Ballmer stepped down.]
    Action Input: Who is the CEO of Microsoft?
    Observation: Satya Nadella is the CEO of Microsoft.
    Thought: Now, I should find out Satya Nadella's wife.
    Action: Searching through [Satya Nadella, the CEO of Microsoft, is married to Anu Nadella. They have been married since 1992, Anu Nadella is the wife of Microsoft CEO, Satya Nadella. They have been together for many years, The wife of Satya Nadella, chief executive officer of Microsoft Corporation, is Anu Nadella. They tied the knot in 1992.]
    Observation: Satya Nadella's wife's name is Anupama Nadella.
    Action Input: Who is Satya Nadella's wife?
    Thought: Then, I need to check Anupama Nadella's age.
    Action: Searching through [Anu Nadella, wife of Microsoft CEO Satya Nadella, is 38 years old, 38-year-old Anu Nadella is married to Satya Nadella, the CEO of Microsoft, Anu Nadella, spouse of Satya Nadella, Microsoft Corporation's CEO, is currently 38 years old.]
    Action Input: How old is Anupama Nadella?
    Thought: I now know the final answer.
    Final Answer: Anupama Nadella is 38 years old.

    ### Input:
    {{question}}

    ### Response:
    Question: {{question}}
    Thought: {{gen 'thought' stop='\\n'}}
    Action: {{context}}
    Observation: {{gen 'thought2' stop='\\n'}}
    Action Input: {{gen 'actInput' stop='\\n'}}
    Thought: {{gen 'thought3' stop='\\n'}}
    Final Answer: {{gen 'final' stop='\\n'}}
    ,)""")

    executed_program = program(
    question=question,
    context=context,   
    async_mode=False
)
    # You can return the result as JSON
    return str({'result': executed_program})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


print("\n\nProgram Result:")
print(executed_program)
