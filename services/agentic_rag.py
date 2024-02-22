from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from loguru import logger
from .vector_store import vector_store


# Create the vector store once
vectordb = vector_store()
chat_history = []

def retriever():
    try:
        logger.info("Retrieving the vectordb for relevant docs...")
        # retrieve documents
        retrieved_docs = vectordb.as_retriever(search_type="mmr", top_k=5)
        return retrieved_docs
    except Exception as e:
        logger.debug(f"Error retrieving the data: {e}")


def compile_and_return_results(query):
    try:
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
        logger.info("Compiling the retrieved documents and passing to LLM...")
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
    except Exception as e:
        logger.debug(f"Encountered error during compilations and passing to LLM. Error: {e}")


# print(compile_and_return_results("What is this document about?")["answer"])
