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
    return response.text

st.set_page_config(page_title="Q&A demo")
st.header("Gemini powered LLM App ")

input=st.text_input("Input:", key="input")
submit=st.button("Ask the questio")
if submit:
    response=get_gemini_response(input)
    st.subheader("The response is ")
    st.write(response.text)
