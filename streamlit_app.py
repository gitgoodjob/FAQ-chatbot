import streamlit as st
import openai
#from langchain.llms import LLaMA
from typing import Optional


def chat_with_gpt(api_key: str, user_input: str) -> str:
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",  # or another model like "gpt-4" if available
        prompt=user_input,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

'''def chat_with_llama(user_input: str) -> str:
    llm = LLaMA(model_path="path/to/llama/model")
    response = llm(user_input)
    return response.strip()'''

def chatbot_interface(api_key: Optional[str], model_choice: str, user_input: str) -> str:
    if model_choice == "GPT (OpenAI)":
        return chat_with_gpt(api_key, user_input)
    elif model_choice == "LLaMA":
        return chat_with_llama(user_input)
    else:
        return "Model not supported yet."

# Streamlit Interface
st.title('FAQ-Bot')

# Choose the model
model_choice = st.selectbox("Choose the AI model:", ["GPT (OpenAI)", "LLaMA"])

# Input the API key if GPT is chosen
api_key = None
if model_choice == "GPT (OpenAI)":
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# User input
user_input = st.text_area("Ask a question:")

# Display response
if st.button("Get Answer"):
    if model_choice == "GPT (OpenAI)" and not api_key:
        st.warning("Please enter your OpenAI API key.")
    elif user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = chatbot_interface(api_key, model_choice, user_input)
        st.write("**Answer:**", response)
