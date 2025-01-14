from dotenv import load_dotenv
load_dotenv() # Load all the environment variable
import os
import streamlit as st
import google.generativeai as genai
from PIL import Image



genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# function tpo load gemiini ai

model=genai.GenerativeModel("gemini-1.5-pro")
def get_gemini_response(input_text, image, prompt):
    response=model.generate_content([input_text,image[0], prompt])
    return response.text


st.set_page_config(page_title="MultiLanguage Invoice Extractor")
st.header("Gemini powered LLM App ")
input_text = st.text_input("Input Prompt:", key="input")
uploaded_file = st.file_uploader("Choose an Image of invoice", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image=image, caption="Uploaded image", use_column_width=True)

submit=st.button("Tell me about the invoice")

input_prompt="""

you are an expert in understanding invoices. We will upload a image an invoice and you will have to answer
any questions based on the uploaded invoice image.

"""
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts= [

            {
                    "mime_type": uploaded_file.type,
                    "data":bytes_data


            }]
        return image_parts
    else:
       raise FileNotFoundError("No File uploaded")
# If submit button clicked
if submit:
    image_data=input_image_details(uploaded_file)
    response=get_gemini_response(input_prompt,image_data,input_text)
    st.subheader("The response is ")
    st.write(response)




