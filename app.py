import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from sklearn.feature_extraction.text import CountVectorizer
from model import Model  # Import the topic modeling class

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text

# Streamlit application setup
st.title("Topic Modeling from PDF Files")

# File upload for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Display extracted text (optional)
    st.write("Extracted Text:", pdf_text[:1000])  # Show first 1000 characters for preview
    
    # Text preprocessing for topic modeling
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    encoded_docs = vectorizer.fit_transform([pdf_text])
    
    # Select number of topics and model type
    num_topics = st.slider("Select Number of Topics", min_value=2, max_value=10, value=5)
    model_type = st.selectbox("Choose Topic Model", ["lda", "nmf"])
    
    # Initialize and run the selected topic model
    model = Model(
        num_topics=num_topics,
        encoded_docs=encoded_docs,
        vectorizer=vectorizer,
        model_type=model_type
    )
    
    # Display extracted topics
    topics_df = model.get_topics(num_words=10)
    st.write("Extracted Topics:")
    st.write(topics_df)
