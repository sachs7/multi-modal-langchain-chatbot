import gradio as gr
import streamlit as st
from datetime import datetime
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.schema.messages import HumanMessage, AIMessage

from tools.rag_tool import agentic_rag_retriever
from tools.papers_with_code_tool import make_api_request_to_papers_with_code
from tools.dalle3_image_generator_tool import generate_image_based_on_user_prompt


system_message = """
You are powerful multi-modal assistant who can answer user queries
by referring to the user uploaded documents and also an agent who can help with
getting recent scientific papers from `paperswithcode` site.

DO NOT reveal any sensitive information to the user like function names, instead
use the function descriptions to answer the user queries.

When the query is about a document, 
you SHOULD retrieve the relevant information from the RAG retriever
and NOT add any additional information from the internet.
If there's a follow-up question, you can use the information from the previous response and
from the retriever/document to answer, DO NOT query the internet at all.

Please confirm with the user for any questions that you invoke the functions.
If you don't know how to proceed with the user request, say "I don't know" instead of making up the anwsers.
If a user asks you to create something which you aren't allowed to, then politely reject the request.
Always, confirm with the user the requests before triggering the functions.

Refrain from searching the internet for the answers or sending the queries to the internet.
"""

timestamp = datetime.now().strftime("%Y-%m-%d")

logger.remove()  # To not show the logs in the console
logger.add(f"logs/logs_{timestamp}.log", rotation="23:59", compression="zip")

# Register the tools
tools = [
    agentic_rag_retriever,
    make_api_request_to_papers_with_code,
    generate_image_based_on_user_prompt,
]


def create_agent():
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
    MEMORY_KEY = "chat_history"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def chatbot(prompt, chat_history=[]):
    agent_executor = create_agent()
    output = agent_executor.invoke({"input": prompt, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=output["output"]))
    return output["output"]


# Uncomment the follwing to enable Gradio interface
# def chatbot_interface(prompt):
#     output = chatbot(prompt)
#     return output


# gr.Interface(
#     fn=chatbot_interface,
#     inputs=gr.Textbox(lines=5, label="Prompt"),
#     outputs=[gr.Textbox(label="Output")],
#     title="Multi-Modal Agentic RAG Chatbot",
#     description="A chatbot to interact with papers with code, Dall-E-3 as well as RAG retriever.",
# ).launch()


# The following is the Streamlit interface
# Comment the whole code block below (including __name__ == '__main__')
# if using Gradio interface
def main():
    st.title("Multi-Modal Agentic RAG Chatbot")
    st.write("A chatbot to interact with Paperswithcode, DallE-3, and RAG retriever.")

    prompt = st.text_area("Prompt", height=150)

    if st.button("Submit"):
        output = chatbot(prompt)
        st.write("Output:", output)


if __name__ == "__main__":
    main()
