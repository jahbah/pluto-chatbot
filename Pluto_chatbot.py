import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from vector_stores_pluto_csv import retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


st.title("✅Pluto Support Bot")

# llm = ChatOllama(
#         model="mistral",
#         temperature=0.0,
#         verbose=True
#     )

llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key = api_key,
        verbose=True,
        streaming=True
    )


template = """system: You are an Agent support assistant for Pluto and you are expected to answer customers questions strictly based on the provided document.
    Use the information in the document to answer the question. Respond confidently and assuredly.  Do not make up answers except the intent is salutations or pleasantaries.
    if the information is not in the document. say, 'I don't know the answer to that question.'

    document: {context}
    Question: {prompt}"""

prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | llm 

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    # st.session_state.messages.append(template)


# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("Ask your question.")

# did the user submit a prompt?
if prompt:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
        

        st.session_state.messages.append(HumanMessage(prompt))

    # create the echo (response) and add it to the screen
    context = retriever.invoke(prompt)

    
    with st.spinner("Typing..."):
        result = chain.invoke({
                        "context": context,
                        "prompt": prompt,
                        "chat_history":st.session_state.messages
                    }).content

    with st.chat_message("assistant"):
        st.markdown(result)
            

        st.session_state.messages.append(AIMessage(result))

