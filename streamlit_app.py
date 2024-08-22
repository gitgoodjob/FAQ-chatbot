import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on FAQ content."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"Error generating response from GPT model: {str(e)}"

    
    elif model == "LLaMA":
        tokenizer = AutoTokenizer.from_pretrained("huggingface/llama")
        model = AutoModelForCausalLM.from_pretrained("huggingface/llama")
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        prompt = f"Based on the following FAQ content, answer the question: {user_input}\n\nFAQ Content:\n{faq_content}"
        response = llm_pipeline(prompt, max_length=150, do_sample=True, temperature=0.7)
        return response[0]['generated_text'].strip()
    
    elif model == "Gemini":
        # Hypothetical example; replace with actual Gemini model usage if available
        # Assuming Gemini has a similar API to GPT
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": "gemini-1.0",
            "prompt": f"Based on the following FAQ content, answer the question: {user_input}\n\nFAQ Content:\n{faq_content}",
            "max_tokens": 150,
            "temperature": 0.7,
        }
        response = requests.post("https://api.gemini.com/v1/complete", headers=headers, json=data)
        return response.json()["choices"][0]["text"].strip()
    
    return "Model not supported yet."

# Streamlit Interface
st.title("AI-Powered FAQ Chatbot")

# Choose the model
model_choice = st.selectbox("Choose the AI model:", ["GPT (OpenAI)", "LLaMA", "Gemini"])

# Input the API key if GPT or Gemini is chosen
api_key = None
if model_choice in ["GPT (OpenAI)", "Gemini"]:
    api_key = st.text_input("Enter your API Key:", type="password")

# Input the FAQ URL
faq_url = st.text_input("Enter the FAQ page URL:")

# User input
user_input = st.text_area("Ask a question:")

# Display response
if st.button("Get Answer"):
    if model_choice in ["GPT (OpenAI)", "Gemini"] and not api_key:
        st.warning("Please enter your API key.")
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
