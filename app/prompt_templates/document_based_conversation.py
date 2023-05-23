from langchain.prompts import StringPromptTemplate
from app.memory.chroma_memory import Chroma

default_template = """You are a librarian AI who uses document information to answer questions. Documents as formatted as follows: [(Document(page_content="<important context>", metadata={{'source': '<source>'}}), <rating>)] where <important context> is the context, <source> is the source, and <rating> is the rating. 
There can be several documents in a conversation.

To assist me in this task, I have access to a vector database that contains various documents related to different topics. Here are some documents that match your query:

Here are some documents to guide your answer:
{search}


Here is the conversation history. Use it to help you:
{history}

Based on this information, how may I assist you today?

{input}
### Response:"""

context_template = """ You are a copyist monk tasked with rewriting a doc in your answer.

Here is the document to copy:
{search}

You MUST return {search} and nothing more.
### Response:"""

guidance_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction:
    You are a librarian AI who uses document information to answer questions. Documents as formatted as follows: [(Document(page_content="<important context>", metadata='source': '<source>'), <rating>)] where <important context> is the context, <source> is the source, and <rating> is the rating. 
    Strictly use the following format:

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
    ,)"""

# Set up a prompt template
class ConversationWithDocumentTemplate(StringPromptTemplate):
    # The template to use
    template: str = default_template
    document_store: Chroma

    def format(self, **kwargs) -> str:
        print("Entering format f with kwargs: ", kwargs)
        # Set the agent_scratchpad variable to that value
        input_question = kwargs.get("input")
        docs = self.document_store.similarity_search_with_score(
            input_question, top_k_docs_for_context=10
        )
        kwargs["search"] = docs

        return self.template.format(**kwargs)

class ContextFromDocumentTemplate(StringPromptTemplate):
    # The template to use
    template: str = context_template
    document_store: Chroma

    def format(self, **kwargs) -> str:
        print("Entering format f with kwargs: ", kwargs)
        # Set the agent_scratchpad variable to that value
        input_question = kwargs.get("input")
        docs = self.document_store.similarity_search_with_score(
            input_question, top_k_docs_for_context=10
        )
        kwargs["search"] = docs

        return self.template.format(**kwargs)

        


Examples = """Before you start on the conversation, here are a few examples on how to use your tools:

Example 1:
Question: What is the author's name?
Thought: I need to check my long-term memory
Action: SearchLongTermMemory
Action Input: "Author Name"
Observation: "Jack Black"
Thought: I now know the answer.
Final Answer: The author's name is Jack Black.

Example 2:
Question: Who is the author?
Thought: I need to check my long-term memory
Action: SearchLongTermMemory
Action Input: "Author Name"
Observation: I cannot find it.
Thought:  Let's look in my short term memory.
Action: FriendlyDiscussion
Action Input: "Author Name"
Observation: [(Document(PAGE_CONTENT="Written by Jack Black in 2009"))
Final Answer: I now know the author's name is Jack Black.

Example 3:
Question: Hi
Thought: This is not a question.
Action: None
Action Input: None
Observation: None is not a valid tool, try another one.
Thought: This is a question that requires a personal answer.
Final Answer: Hello, friend! How can I help you?

If a tool is not listed above, do not use an action.

Begin!
"""

class guidanceTemplate(StringPromptTemplate):
    # The template to use
    template: str = guidance_template
    document_store: Chroma

    def format(self, **kwargs) -> str:
        print("Entering format f with kwargs: ", kwargs)
        # Set the agent_scratchpad variable to that value
        input_question = kwargs.get("input")
        docs = self.document_store.similarity_search_with_score(
            input_question, top_k_docs_for_context=10
        )
        kwargs["search"] = docs

        return self.template.format(**kwargs)


Examples = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction:
    You are a librarian AI who uses document information to answer questions. Documents as formatted as follows: [(Document(page_content="<important context>", metadata='source': '<source>'), <rating>)] where <important context> is the context, <source> is the source, and <rating> is the rating. 
    Strictly use the following format:

    Question: How old is CEO of Microsoft wife?
    Thought: First, I need to find who is the CEO of Microsoft.
    Action: Searching through [(Document(page_content='Satya Nadella is the CEO of Microsoft Corporation. He took over as CEO in February 2014, succeeding Steve Ballmer.'), 0.95), (Document(page_content='Microsoft, the Redmond-based tech giant, is led by CEO Satya Nadella, who assumed the role in 2014.'), 0.91), (Document(page_content='The chief executive officer of Microsoft Corporation is Satya Nadella. Nadella took the helm in 2014 after Steve Ballmer stepped down.'), 0.93)]
    Action Input: Who is the CEO of Microsoft?
    Observation: Satya Nadella is the CEO of Microsoft.
    Thought: Now, I should find out Satya Nadella's wife.
    Action: Searching through [(Document(page_content='Satya Nadella, the CEO of Microsoft, is married to Anu Nadella. They have been married since 1992.'), 0.96),(Document(page_content='Anu Nadella is the wife of Microsoft CEO, Satya Nadella. They have been together for many years.'), 0.94),(Document(page_content='The wife of Satya Nadella, chief executive officer of Microsoft Corporation, is Anu Nadella. They tied the knot in 1992.'}), 0.95)]
    Observation: Satya Nadella's wife's name is Anupama Nadella.
    Action Input: Who is Satya Nadella's wife?
    Thought: Then, I need to check Anupama Nadella's age.
    Action: Searching through "[(Document(page_content='Anu Nadella, wife of Microsoft CEO Satya Nadella, is 38 years old.), 0.95)]
    Action Input: How old is Anupama Nadella?
    Thought: I now know the final answer.
    Final Answer: Anupama Nadella is 38 years old."""


