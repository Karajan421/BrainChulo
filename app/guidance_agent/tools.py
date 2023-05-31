import os
import re

from colorama import Fore
from colorama import Style
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return text


def load_unstructured_document(document: str) -> list[Document]:
    with open(document, "r") as file:
        text = file.read()
    title = os.path.basename(document)
    return [Document(page_content=text, metadata={"title": title})]


def split_documents(
    documents: list[Document], chunk_size: int = 100, chunk_overlap: int = 0
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def checkQuestion(llm, question: str, retriever):
    QUESTION_CHECK_PROMPT_TEMPLATE = """You MUST answer with 'yes' or 'no'. Given the following pieces of context, determine if there are any elements related to the question in the context.
Don't forget you MUST answer with 'Yes' or 'No'.
Context:{context}
Question: Are there any elements related to ""{question}"" in the context?
"""

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    answer_data = qa({"query": question})
    if "result" not in answer_data:
        print(f"\033[1;31m{answer_data}\033[0m")
        return "Issue in retrieving the answer."

    context_documents = answer_data["source_documents"]
    context = " ".join([clean_text(doc.page_content) for doc in context_documents])
    question_check_prompt = QUESTION_CHECK_PROMPT_TEMPLATE.format(
        context=context, question=question
    )
    print(Fore.GREEN + Style.BRIGHT + question_check_prompt + Style.RESET_ALL)
    answerable = llm(question_check_prompt)
    print(Fore.GREEN + Style.BRIGHT + answerable + Style.RESET_ALL)
    #return answerable[-3:]
    if "yes" in answerable.lower():
        return "Yes"
    else:
        return " No"


def load_tools(llm_model, settings, filepath=False):
    print(Fore.GREEN + Style.BRIGHT + f"Starting reviewing tools...." + Style.RESET_ALL)
    
    llm_pipe = pipeline(
        task="text2text-generation",
        model='lmsys/fastchat-t5-3b-v1.0',
        model_kwargs={}
    )
    llm = HuggingFacePipeline(pipeline=llm_pipe)

    if filepath:
        def ingest_file(file_path_arg):
            documents = load_unstructured_document(file_path_arg)
            documents = split_documents(documents, chunk_size=100, chunk_overlap=20)
            EmbeddingsModel = settings.embeddings_map.get(settings.embeddings_model)
            if EmbeddingsModel is None:
                raise ValueError(f"Invalid embeddings model: {settings.embeddings_model}")

            model_kwargs = (
                {"device": "cuda:0"}
                if EmbeddingsModel == HuggingFaceInstructEmbeddings
                else {}
            )
            embedding = EmbeddingsModel(
                model_name=settings.embeddings_model, model_kwargs=model_kwargs
            )
            vectordb = Chroma.from_documents(documents=documents, embedding=embedding)
            retriever = vectordb.as_retriever(search_kwargs={"k": 4})

            return retriever, file_path_arg

        retriever, _ = ingest_file(filepath)

    def searchChroma(key_word):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )
        res = qa.run(key_word)
        return res

    dict_tools = {
        "Chroma Search": searchChroma,
        "Check Question": lambda question: checkQuestion(
            llm, question, retriever
        ),
    }
    if filepath:
        dict_tools["File Ingestion"] = ingest_file

    return dict_tools
