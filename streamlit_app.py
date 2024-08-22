import openai
import streamlit as st
import requests
from google.generativeai import Client as GeminiClient

# Function to fetch and process FAQ content from a URL
def fetch_faq_content(faq_url: str) -> str:
    try:
        response = requests.get(faq_url)
        response.raise_for_status()
        return response.text  # Parse as needed (HTML, JSON, etc.)
    except Exception as e:
        return f"Error fetching FAQ content: {str(e)}"

# Function to handle AI model interaction
def chat_with_faq(faq_content: str, model: str, api_key: str, user_input: str) -> str:
    if model == "GPT (OpenAI)":
        openai.api_key = api_key
        prompt = f"Based on the following FAQ content, answer the question: {user_input}\n\nFAQ Content:\n{faq_content}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on FAQ content."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error generating response from GPT model: {str(e)}"

    elif model == "Gemini":
        client = GeminiClient(api_key=api_key)
        prompt = f"FAQ Content:\n{faq_content}\nQuestion: {user_input}"
        try:
            response = client.generate_text(prompt=prompt)
            return response['text'].strip()
        except Exception as e:
            return f"Error generating response from Gemini model: {str(e)}"

    else:
        return "Unsupported model selected."

# Streamlit interface
st.title("AI-Powered FAQ Chatbot")

# User input fields
model_choice = st.selectbox("Choose the AI model", ["GPT (OpenAI)", "Gemini"])
api_key = st.text_input("Enter your API key", type="password")
faq_url = st.text_input("Enter the FAQ URL")

if st.button("Fetch FAQ"):
    faq_content = fetch_faq_content(faq_url)
    st.text_area("FAQ Content", faq_content)

user_input = st.text_input("Ask a question based on the FAQ")

if st.button("Submit"):
    if faq_url and faq_content and user_input:
        response = chat_with_faq(faq_content, model_choice, api_key, user_input)
        st.write(f"Answer: {response}")
    else:
        st.write("Please provide the FAQ URL, fetch FAQ, and enter your question.")
