import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

import os

os.environ["GROQ_API_KEY"] = st.secrets["OpenAI_key"]

# Creating an Array for storing the chat history for the model.
context = []


# Set the title of the Streamlit app
st.set_page_config(layout="wide")#, page_title="Llama 3 Model Document Q&A"
st.title("LANG-CHAIN CHAT WITH LLAMA")

# Creating a Session State array to store and show a copy of the conversation to the user.
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Create the Sidebar
sidebar = st.sidebar

# Create the reset button for the chats
clear_chat = sidebar.button("Clear Chat")
if clear_chat:
    context = []
    st.session_state.messages =[]



# Defining out LLM Model
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # streaming=True
)

# Create the function to stream output from llm
def get_response(question, context):

    # Define the prompt template
    prompt_template = """You are a helpful assistant. You should answer the following question or questions as best you can.

    Chat history:{context}

    Question: {question}
    answer"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


    # Create the chain
    chain = PROMPT | llm | StrOutputParser()

    # Return a streaming output.
    return chain.stream({
        "context": context,
        "question": question,
    })

# ------------------------------------------------------------------------------------------------------------------------------

def start_app():



        question = st.chat_input("Ask Anything.", key=1)

        if question:


            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])


            st.session_state.messages.append({"role": "user", "content": question})
    
            # response = get_response(PROMPT, llm).invoke({"input": question, "context": context})
            with st.chat_message("Human"):
                st.markdown(question)

            with st.chat_message("AI"):
                response = st.write_stream(get_response(question, st.session_state.messages))


            st.session_state.messages.append({"role": "assistant", "content": response})


            for message in st.session_state.messages:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)

            context.append(HumanMessage(content=question))
            context.append(AIMessage(content=str(response)))



if __name__ == "__main__":
    start_app()