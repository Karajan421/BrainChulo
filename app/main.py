import os
import shutil
from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, create_engine, Session
from models.all import Conversation, Message
from typing import List
from conversations.document_based import DocumentBasedConversation
from datetime import datetime
from settings import load_config, logger
import nest_asyncio

config = load_config()

sqlite_database_url = "sqlite:///data/brainchulo.db"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_database_url, echo=True, connect_args=connect_args)

convo = DocumentBasedConversation()

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://0.0.0.0:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.post("/conversations", response_model=Conversation)
def create_conversation(*, session: Session = Depends(get_session), conversation: Conversation):
    """
    Create a new conversation.
    """
    conversation = Conversation.from_orm(conversation)
    session.add(conversation)
    session.commit()
    session.refresh(conversation)

    return conversation


@app.get("/conversations", response_model=List[Conversation])
def get_conversations(session: Session = Depends(get_session)):
    """
    Get all conversations.
    """
    return session.query(Conversation).all()


@app.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: int, session: Session = Depends(get_session)):
    """
    Get a conversation by id.
    """
    conversation = session.get(Conversation, conversation_id)

    return conversation

@app.post("/conversations/{conversation_id}/messages", response_model=Message)
def create_message(*, session: Session = Depends(get_session), conversation_id: int, message: Message):
    """
    Create a new message.
    """
    message = Message.from_orm(message)
    session.add(message)
    session.commit()
    session.refresh(message)

    return message

@app.post("/conversations/{conversation_id}/files", response_model=dict)
def upload_file(*, conversation_id: int, file: UploadFile):
    """
    Upload a file.
    """
    try:
        uploaded_file_name = file.filename
        filepath = os.path.join(
            os.getcwd(), "data", config.upload_path, uploaded_file_name
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)

        convo.load_document(filepath, conversation_id)

        return {"text": f"{uploaded_file_name} has been loaded into memory for this conversation."}
    except Exception as e:
        logger.error(f"Error adding file to history: {e}")
        return f"Error adding file to history: {e}"

@app.post('/llm', response_model=str)
def llm(*, query: str):
    """
    Query the LLM
    """
    return convo.predict(query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=7865, reload=True)