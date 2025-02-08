import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 200)
        st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        st.session_state.vector = FAISS.from_documents(st.session_state.final_doc, st.session_state.embeddings)

prompt1 = st.text_input("Enter your question from the documents")


if st.button("Create Vector store"):
    vector_embedding()
    st.write("Vector store db is ready!")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input" : prompt1})   
    print("Response time :",time.process_time()-start)
    st.write(response["answer"])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------")
