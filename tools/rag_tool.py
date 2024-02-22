from langchain.agents import tool

from services.agentic_rag import compile_and_return_results

@tool
def agentic_rag_retriever(question: str):
    """
    Use the docs in the documents_directory to create a vector store and retriever
    and answer the questions asked by the user.
    """
    return compile_and_return_results(question)["answer"]
