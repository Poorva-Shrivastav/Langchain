import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq #for chat window
from langchain_community.document_loaders import WebBaseLoader #for scraping
from langchain_community.embeddings import OllamaEmbeddings #embedding
from langchain.text_splitter import RecursiveCharacterTextSplitter #splitting the docs scraped from the website
from langchain.chains.combine_documents import create_stuff_documents_chain #chain for passing list of docs to model
from langchain_core.prompts import ChatPromptTemplate #create custom prompt template
from langchain.chains.retrieval import create_retrieval_chain #retrieves the docs from the chain
from langchain_community.vectorstores import FAISS
import time

load_dotenv()

#load groq api key
groq_api_key = os.environ['GROQ_API_KEY']


if "vector" not in st.session_state:
    st.session_state.embedding = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://www.python.org/doc/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 200)
    st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])

    st.session_state.vector = FAISS.from_documents(st.session_state.final_doc, st.session_state.embedding)

st.title("Chat Groq Demo")
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide most accurate response based on the question
<context>
{context}
<context>
Question : {input}                                          
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here: ")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input" : prompt})
    print(f"Response time:  {time.process_time() - start}")
    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------")






















