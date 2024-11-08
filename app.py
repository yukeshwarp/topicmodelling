import streamlit as st
import requests
import logging
import time
import random
from PyPDF2 import PdfReader
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Azure OpenAI API configuration
azure_endpoint = "https://uswest3daniel.openai.azure.com"
model = "GPT-4Omni"  # Replace with your actual model
api_version = "2024-10-01-preview"
api_key = "fcb2ce5dc289487fad0f6674a0b35312" # Replace with your Azure API key
HEADERS = {"Content-Type": "application/json", "api-key": api_key}

# Set up logging
logging.basicConfig(level=logging.INFO)

def preprocess_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def summarize_page_for_topics(page_text, previous_summary, page_number, system_prompt,
                              max_retries=5, base_delay=1, max_delay=32):
    preprocessed_page_text = preprocess_text(page_text)
    preprocessed_previous_summary = preprocess_text(previous_summary)

    # Combine current page text with previous summary
    prompt_message = (
        f"Please rewrite the following page content from (Page {page_number}) along with context from the previous page summary "
        f"to make them concise and well-structured. Maintain proper listing and referencing of the contents if present. "
        f"Do not add any new information or make assumptions. Keep the meaning accurate and the language clear.\n\n"
        f"Previous page summary: {preprocessed_previous_summary}\n\n"
        f"Current page content:\n{preprocessed_page_text}\n"
    )

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_message},
        ],
        "temperature": 0.5,
    }

    attempt = 0
    while attempt < max_retries:
        try:
            # Sending request to Azure OpenAI API for summarization
            response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=HEADERS,
                json=data,
                timeout=50,
            )
            response.raise_for_status()
            logging.info(f"Summary retrieved for page {page_number}")
            return (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No summary provided.")
                .strip()
            )

        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                logging.error(f"Error summarizing page {page_number}: {e}")
                return f"Error: Unable to summarize page {page_number} due to network issues or API error."

            delay = min(max_delay, base_delay * (2 ** attempt))
            jitter = random.uniform(0, delay)
            logging.warning(f"Retrying in {jitter:.2f} seconds (attempt {attempt}) due to error: {e}")
            time.sleep(jitter)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text_by_page = []
    for page_num in range(len(reader.pages)):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            text_by_page.append(page_text)
    return text_by_page

# Topic Modeling with NMF
def extract_topics_from_text(texts, num_topics=5, num_words=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    nmf = NMF(n_components=num_topics, random_state=42)
    nmf.fit(X)
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[:-num_words - 1:-1]
        top_words = feature_names[top_words_idx]
        topics.append(" ".join(top_words))
    
    return topics, nmf.transform(X)

def plot_topic_distribution(topic_probabilities, num_topics):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(num_topics), topic_probabilities, color='skyblue')
    ax.set_xlabel('Topic')
    ax.set_ylabel('Probability')
    ax.set_title('Topic Distribution across Pages')
    st.pyplot(fig)

# Streamlit app UI
st.title("Hybrid Summarization and Topic Extraction App")

uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
system_prompt = st.text_area("Enter system prompt for the model:", 
                             "Summarize the document contents in a concise, structured manner.")

if uploaded_pdf:
    # Extract text from uploaded PDF
    pdf_pages = extract_text_from_pdf(uploaded_pdf)
    st.write("### Extracted Topics and Summaries by Page")
    previous_summary = ""  # Initialize previous summary as empty

    # Collect text for topic modeling
    full_text = []
    for page_number, page_text in enumerate(pdf_pages, start=1):
        with st.spinner(f"Summarizing page {page_number}..."):
            page_summary = summarize_page_for_topics(
                page_text,
                previous_summary,
                page_number,
                system_prompt
            )
            st.subheader(f"Page {page_number}")
            st.write(page_summary)
            previous_summary = page_summary  # Update previous summary for context in next page
            full_text.append(page_text)  # Add page text for topic modeling
    
    st.success("All pages summarized successfully.")
    
    # Perform topic modeling on the entire document
    topics, topic_probabilities = extract_topics_from_text(full_text, num_topics=5, num_words=10)
    
    st.write("### Identified Topics")
    for idx, topic in enumerate(topics):
        st.write(f"**Topic {idx+1}:** {topic}")
    
    # Plot topic distribution across the document
    plot_topic_distribution(topic_probabilities.mean(axis=0), num_topics=5)
