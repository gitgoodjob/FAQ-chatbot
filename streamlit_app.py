import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup

def extract_faq_from_url(faq_url: str) -> str:
    """Extracts FAQ content from a given URL."""
    try:
        response = requests.get(faq_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Example logic to extract FAQ based on common HTML structure
        faq_sections = soup.find_all('div', class_='faq-section')  # Example class
        faq_text = ""

        for section in faq_sections:
            questions = section.find_all('h3')
            answers = section.find_all('p')
            for question, answer in zip(questions, answers):
                faq_text += f"Q: {question.get_text(strip=True)}\nA: {answer.get_text(strip=True)}\n\n"

        return faq_text.strip()
    except Exception as e:
        return f"Error extracting FAQs: {str(e)}"

def chat_with_faq(faq_content: str, model: str, api_key: str, user_input: str) -> str:
    """Uses the AI model to answer based on the provided FAQ content."""
    if model == "GPT (OpenAI)":
        openai.api_key = api_key
        prompt = f"Based on the following FAQ content, answer the question: {user_input}\n\nFAQ Content:\n{faq_content}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    # Placeholder for Gemini or other models
    elif model == "Gemini":
        # Add Gemini-specific code here
        return "Gemini model not yet implemented."
    # Add logic for LLaMA and others as needed
    return "Model not supported yet."

# Streamlit Interface
st.title("AI-Powered FAQ Chatbot")

# Choose the model
model_choice = st.selectbox("Choose the AI model:", ["GPT (OpenAI)", "Gemini", "LLaMA"])

# Input the API key if GPT is chosen
api_key = None
if model_choice == "GPT (OpenAI)":
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Input the FAQ URL
faq_url = st.text_input("Enter the FAQ page URL:")

# User input
user_input = st.text_area("Ask a question:")

# Display response
if st.button("Get Answer"):
    if model_choice == "GPT (OpenAI)" and not api_key:
        st.warning("Please enter your OpenAI API key.")
    elif not faq_url.strip():
        st.warning("Please enter a valid FAQ page URL.")
    elif user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        faq_content = extract_faq_from_url(faq_url)
        if "Error" in faq_content:
            st.error(faq_content)
        else:
            response = chat_with_faq(faq_content, model_choice, api_key, user_input)
            st.write("**Answer:**", response)
