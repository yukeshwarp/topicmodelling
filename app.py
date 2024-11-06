import streamlit as st
import pandas as pd
from model import Model  # Import the modified model class here

# Streamlit UI for topic modeling app
st.title("Topic Modeling Comparison App")

# Parameters for topic modeling
num_topics = st.sidebar.number_input("Number of Topics", min_value=2, max_value=50, value=10)
model_type = st.sidebar.selectbox("Model Type", ["lda", "nmf", "ctm"])
num_words = st.sidebar.number_input("Number of Words per Topic", min_value=2, max_value=20, value=10)

# File upload
uploaded_file = st.file_uploader("Upload a text file", type="csv")
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    if 'text' not in data.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        docs = data['text'].values.tolist()
        filtered_lemmas = [doc.split() for doc in docs]  # Simple tokenizer for demo purposes

        # Instantiate model
        st.write(f"Training {model_type.upper()} model with {num_topics} topics...")
        model = Model(num_topics=num_topics, docs=docs, filtered_lemmas=filtered_lemmas, model_type=model_type)
        
        # Display topics
        topics = model.get_topics(num_words=num_words)
        st.write("Generated Topics:")
        for idx, row in topics.iterrows():
            st.write(f"**Topic {row['Topic']}**: {', '.join([word for word, _ in row['Words']])}")
