
from dotenv import load_dotenv
load_dotenv() # Load all the environment variable
import os
import streamlit as st
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# function tpo load gemiini ai

model=genai.GenerativeModel("gemini-pro")
def get_gemini_response(question):
    response=model.generate_content(question)
    return response

if "chat_history" not in st.session_state:
    st.session_state['chat_history']=[]

# Creating header and page configuration
st.set_page_config(page_title="Chatbot  demo")
st.header("Gemini powered Chatbot ")
# To read input from text box
input_text=st.text_input("Input:", key="input")
submit=st.button("Ask the question")
if submit and input_text:
    response=get_gemini_response(input_text)
    st.session_state['chat_history'].append(('you', input_text))
    st.subheader("The response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(('Bot',chunk.text))
    st.subheader('The chat history is ')

    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

