import streamlit as st
import google.generativeai as genai

# Configure API
genai.configure(api_key="AIzaSyCqWdsH08YeGqmHnmeJqy7aZTpdjDDtpjU")

# Load model
model = genai.GenerativeModel("gemini-1.5-flash")

# UI
st.title("AI Customer Support Chatbot")

user_input = st.text_input("Ask something:")

if user_input:
    response = model.generate_content(user_input)
    st.write(response.text)