#gemma2-9b-it
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load all the environment variables
import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
import time


groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
st.title("Gemma Model Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
print("llm read")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question

<context>
{context}
<context>
Question:{input}
"""
)

def vector_embedding():
    if "vectors"  not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        print("embedding loaded")
        st.session_state.loader=PyPDFDirectoryLoader("./cricket") # Data Ingestion
        print("Data Ingestion Done")

        st.session_state.docs=st.session_state.loader.load() # Docs loading
        print("Docs Loaded")
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
        st.session_state.chunks=st.session_state.text_splitter.split_documents(st.session_state.docs)
        print("Chunks created")
        st.session_state.vectors=FAISS.from_documents(st.session_state.chunks,st.session_state.embeddings)
        print("vectors created")

prompt1=st.text_input("What you want to ask from docs ?")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector store DB is ready")

     
if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    print("retriever chain created")

    start= time.process_time()
    response=retrieval_chain.invoke({"input":prompt1})
    print("response generated")
    st.write(response['answer'])


    with st.expander('Documetn Similatity Search'):
    # Find relevan chunks 
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")