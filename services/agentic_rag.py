import os
import shutil
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from loguru import logger

# Directories
documents_directory = "docs"
persist_directory = "chroma_persists"

chat_history = []
timestamp = datetime.now().strftime("%Y-%m-%d")

logger.remove()  # To not show the logs in the console
logger.add(f"logs/logs_{timestamp}.log", rotation="23:59", compression="zip")


# Function to create the directory if it doesn't exist
def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as e:
        logger.debug(f"Error creating a directory: {e}")


# Function to remove the directory if it exist
def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        # Folder and its content removed
        logger.info("Folder and its content removed")
    except Exception as e:
        logger.debug(f"Error removing the folder {folder_path}: {e}")


# Document loader
def document_loader():
    try:
        logger.info("Loading documents")
        loader = PyPDFDirectoryLoader(documents_directory)
        docs = loader.load()
        return docs
    except Exception as e:
        logger.debug(f"Error loading the documents: {e}")


def text_splitter():
    # remove old database files if any
    remove_folder(persist_directory)

    # Create the directory if it doesn't exist
    create_directory(persist_directory)

    documents = document_loader()
    logger.info("-" * 50)
    logger.info(f"Length of documents: {len(documents)}")
    try:
        # split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(
            f"Number of documents after passing through text_splitter: {len(split_docs)}"
        )
        return split_docs
    except Exception as e:
        logger.debug(f"Error indexing the data: {e}")


def vector_store():
    split_docs = text_splitter()
    try:
        # define embedding
        embedding = OpenAIEmbeddings()
        # create vector database from data
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding,
            persist_directory=persist_directory,
        )
        logger.info(f"VectorDB collection count: {vectordb._collection.count()}")
        vectordb.persist()
        return vectordb
    except Exception as e:
        logger.debug(f"Error indexing the data: {e}")

# Create the vector store once
vectordb = vector_store()

def retriever():
    try:
        # retrieve documents
        retrieved_docs = vectordb.as_retriever(search_type="mmr", top_k=5)
        return retrieved_docs
    except Exception as e:
        logger.debug(f"Error retrieving the data: {e}")


def compile_and_return_results(query):
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end.
    Answer only if the question is within the context, if anything outside the context, then don't answer.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible and to the given context. Do NOT query the internet for answers.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate.from_template(template)

    retrieved_docs = retriever()
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0),
        chain_type="stuff",
        retriever=retrieved_docs,
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

    result = qa({"question": query, "chat_history": chat_history})
    return result


# print(compile_and_return_results("What is this document about?")["answer"])
