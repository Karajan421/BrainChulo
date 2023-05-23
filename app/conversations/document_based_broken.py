from langchain.document_loaders import TextLoader
from app.memory.chroma_memory import Chroma
from langchain.memory import VectorStoreRetrieverMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.agents import Tool, initialize_agent, AgentType, load_tools
from langchain.schema import OutputParserException
from app.llms.oobabooga_llm import OobaboogaLLM
from app.prompt_templates.document_based_conversation import (
    Examples,
    ConversationWithDocumentTemplate,
    guidanceTemplate
)
from app.settings import logger, load_config
import guidance
config = load_config()

from app.guidance.llms._text_generation_web_ui import TextGenerationWebUI
guidance.llm = TextGenerationWebUI()
guidance.llm.caching = False

USE_AGENT = config.use_agent


class DocumentBasedConversation:
    def __init__(self):
        """
        Initializes an instance of the class. It sets up LLM, text splitter, vector store, prompt template, retriever,
        conversation chain, tools, and conversation agent if USE_AGENT is True.
        """

        self.llm = OobaboogaLLM()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=20, length_function=len)
        self.vector_store_docs = Chroma(collection_name="docs_collection")
        self.vector_store_convs = Chroma(collection_name="convos_collection")

        convs_retriever = self.vector_store_convs.get_store().as_retriever(
            search_kwargs=dict(top_k_docs_for_context=10)
        )

        convs_memory = VectorStoreRetrieverMemory(retriever=convs_retriever)

        self.prompt = ConversationWithDocumentTemplate(
            input_variables=["input", "history"],
            document_store=self.vector_store_docs,
        )

        self.guidance_prompt = guidanceTemplate(
            input_variables=["input", "history"],
            document_store=self.vector_store_docs,
        )

        self.conversation_chain = ConversationChain(
            llm=self.llm, prompt=self.prompt, memory=convs_memory, verbose=True
        )
        self.program = guidance("""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction:
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
    Action: Searching through [(Document(page_content='Satya Nadella is the CEO of Microsoft Corporation. He took over as CEO in February 2014, succeeding Steve Ballmer.', metadata={'source': '/home/user/Documents/microsoft_info.txt'}), 0.95), (Document(page_content='Microsoft, the Redmond-based tech giant, is led by CEO Satya Nadella, who assumed the role in 2014.', metadata={'source': '/home/user/Documents/tech_leaders.txt'}), 0.91), (Document(page_content='The chief executive officer of Microsoft Corporation is Satya Nadella. Nadella took the helm in 2014 after Steve Ballmer stepped down.', metadata={'source': '/home/user/Documents/business_leaders.txt'}), 0.93)]
    Action Input: Who is the CEO of Microsoft?
    Observation: Satya Nadella is the CEO of Microsoft.
    Thought: Now, I should find out Satya Nadella's wife.
    Action: Searching through [(Document(page_content='Satya Nadella, the CEO of Microsoft, is married to Anu Nadella. They have been married since 1992.', metadata={'source': '/home/user/Documents/microsoft_info.txt'}), 0.96),(Document(page_content='Anu Nadella is the wife of Microsoft CEO, Satya Nadella. They have been together for many years.', metadata={'source': '/home/user/Documents/tech_leaders_families.txt'}), 0.94),(Document(page_content='The wife of Satya Nadella, chief executive officer of Microsoft Corporation, is Anu Nadella. They tied the knot in 1992.', metadata={'source': '/home/user/Documents/business_leaders_families.txt'}), 0.95)]
    Observation: Satya Nadella's wife's name is Anupama Nadella.
    Action Input: Who is Satya Nadella's wife?
    Thought: Then, I need to check Anupama Nadella's age.
    Action: Searching through [(Document(page_content='Anu Nadella, wife of Microsoft CEO Satya Nadella, is 38 years old.', metadata={'source': '/home/user/Documents/microsoft_info.txt'}), 0.96),(Document(page_content='38-year-old Anu Nadella is married to Satya Nadella, the CEO of Microsoft.', metadata={'source': '/home/user/Documents/tech_leaders_families.txt'}), 0.94), (Document(page_content='Anu Nadella, spouse of Satya Nadella, Microsoft Corporation's CEO, is currently 38 years old.', metadata={'source': '/home/user/Documents/business_leaders_families.txt'}), 0.95)]
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
   
        if USE_AGENT:
            tools = load_tools([])

            tools.append(
                Tool(
                    name="FriendlyDiscussion",
                    func=self.conversation_chain.run,
                    description="useful when you need to discuss with a human based on relevant context from previous conversation",
                )
            )

            tools.append(
                Tool(
                    name="SearchLongTermMemory",
                    func=self.search,
                    description="useful when you need to search for information in long-term memory",
                )
            )

            self.conversation_agent = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=convs_memory,
                verbose=True,
            )

    def load_document(self, document_path, conversation_id=None):
        """
        Load a document from a file and add its contents to the vector store.

        Args:
          document_path: A string representing the path to the document file.

        Returns:
          None.
        """
        text_loader = TextLoader(document_path, encoding="utf8")
        documents = text_loader.load()
        documents = self.text_splitter.split_documents(documents)

        if conversation_id is not None:
            for doc in documents:
                doc.metadata["conversation_id"] = conversation_id

        self.vector_store_docs.add_documents(documents)

    def search(self, search_input, conversation_id=None):
        """
        Search for the given input in the vector store and return the top 10 most similar documents with their scores.
        This function is used as a helper function for the SearchLongTermMemory tool

        Args:
          search_input (str): The input to search for in the vector store.

        Returns:
          List[Tuple[str, float]]: A list of tuples containing the document text and their similarity score.
        """
        if conversation_id is not None:
            filter = {"conversation_id": conversation_id}
        else:
            filter = {}

        logger.info(f"Searching for: {search_input} in LTM")
        docs = self.vector_store_docs.similarity_search_with_score(
            search_input, k=5, filter=filter
        )
        return docs

    def predict(self, input):

        """
        Predicts a response based on the given input.

        Args:
          input (str): The input string to generate a response for.

        Returns:
          str: The generated response string.

        Raises:
          OutputParserException: If the response from the conversation agent could not be parsed.
        """
        if USE_AGENT:
            try:
                response = self.conversation_agent.run(
                    input=f"{Examples}\n{input}",
                )
            except OutputParserException as e:
                response = str(e)
                if not response.startswith("Could not parse LLM output: `"):
                    raise e
                response = response.removeprefix(
                    "Could not parse LLM output: `"
                ).removesuffix("`")
        else:
            response = self.conversation_chain.predict(input="input")

        return response

    def guided_predict(self, input):
        """
        Predicts a response based on the given input using the guidance library.

        Args:
        input (str): The input string to generate a response for.

        Returns:
        str: The generated response string.
        """
        #context = self.search(input)
        executed_program = self.program(
                question=input,
                context=(self.vector_store_docs.similarity_search_with_score(
            input, k=5, filter=filter
        )),
                async_mode=False
            )

        return executed_program
