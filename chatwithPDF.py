from dotenv import load_dotenv
load_dotenv()  # Load all the environment variables
import os
import streamlit as st
import google.generativeai as genai
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini AI model
model = genai.GenerativeModel("gemini-1.5-pro")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    save_path = os.path.join(os.getcwd(), "faiss_index")
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory created at {save_path}")
    
    vector_store.save_local(save_path)
    print(f"Vector store saved at {save_path}")
    
    # Verify if the file exists after saving
    index_file = os.path.join(save_path, 'index.faiss')
    if os.path.exists(index_file):
        print(f"Index file successfully saved at {index_file}")
    else:
        print(f"Failed to save index file at {index_file}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    make sure to provide all the details. If the answer is not in the provided context, just say,
    "answer is not available in the context," and do not provide the wrong answer.
    Context :\n{context}\n
    Question: \n{question}\n
    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    save_path = os.path.join(os.getcwd(), 'faiss_index')
    
    # Add a check to ensure the file exists before loading
    if not os.path.exists(os.path.join(save_path, 'index.faiss')):
        raise FileNotFoundError(f"Index file not found at {save_path}")
    
    new_db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with multiple PDF")
    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Ask a question from PDF files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files and click on button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                print(raw_text[0])
                text_chunks = get_text_chunks(raw_text)
                print(text_chunks[0])
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
