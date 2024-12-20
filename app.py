import streamlit as st
import requests
import logging
import time
import random
from PyPDF2 import PdfReader
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def extractive_summarization(page_text, page_number, system_prompt, max_retries=5, base_delay=1, max_delay=32):
    prompt_message = (
        f"Please extract the key points from the following content on Page {page_number}. "
        f"Maintain the original meaning and structure, but only include the most important information.\n\n"
        f"Content: {page_text}\n"
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
            response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=HEADERS,
                json=data,
                timeout=50,
            )
            response.raise_for_status()
            logging.info(f"Extracted key points for page {page_number}")
            return (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No extractive summary provided.")
                .strip()
            )

        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                logging.error(f"Error in extractive summarization for page {page_number}: {e}")
                return f"Error: Unable to extract key points for page {page_number} due to network issues or API error."

            delay = min(max_delay, base_delay * (2 ** attempt))
            jitter = random.uniform(0, delay)
            logging.warning(f"Retrying in {jitter:.2f} seconds (attempt {attempt}) due to error: {e}")
            time.sleep(jitter)

def abstractive_summarization(page_text, previous_summary, page_number, system_prompt, max_retries=5, base_delay=1, max_delay=32):
    preprocessed_page_text = preprocess_text(page_text)
    preprocessed_previous_summary = preprocess_text(previous_summary)

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

def plot_confusion_matrix_like_heatmap(topic_probabilities):
    topic_prob_df = pd.DataFrame(topic_probabilities, columns=[f'Topic {i+1}' for i in range(topic_probabilities.shape[1])])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(topic_prob_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Topic Distribution across Pages (Confusion Matrix Style)")
    plt.xlabel("Topics")
    plt.ylabel("Pages")
    st.pyplot(plt.gcf())

# Streamlit app UI
st.title("Hybrid Summarization and Topic Extraction App")

uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
system_prompt = st.text_area("Enter system prompt for the model:", 
                             "Summarize the document contents in a concise, structured manner.")

if uploaded_pdf:
    pdf_pages = extract_text_from_pdf(uploaded_pdf)
    st.write("### Extracted Topics and Summaries by Page")
    previous_summary = ""

    full_text = []
    for page_number, page_text in enumerate(pdf_pages, start=1):
        with st.spinner(f"Summarizing page {page_number}..."):
            extractive_summary = extractive_summarization(
                page_text,
                page_number,
                system_prompt
            )
            abstractive_summary = abstractive_summarization(
                page_text,
                previous_summary,
                page_number,
                system_prompt
            )
            st.subheader(f"Page {page_number}")
            st.write("### Extractive Summary")
            st.write(extractive_summary)
            st.write("### Abstractive Summary")
            st.write(abstractive_summary)
            previous_summary = abstractive_summary
            full_text.append(page_text)
    
    st.success("All pages summarized successfully.")
    
    topics, topic_probabilities = extract_topics_from_text(full_text, num_topics=5, num_words=10)
    
    st.write("### Identified Topics")
    for idx, topic in enumerate(topics):
        st.write(f"**Topic {idx+1}:** {topic}")
    
    plot_topic_distribution(topic_probabilities.mean(axis=0), num_topics=5)
    
    st.write("### Topic Distribution across Pages (Confusion Matrix Style)")
    plot_confusion_matrix_like_heatmap(topic_probabilities)
