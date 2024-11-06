import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from model import Model
from sklearn.feature_extraction.text import CountVectorizer

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
    
    if not pdf_text.strip():
        st.error("The PDF does not contain any extractable text.")
    else:
        # Text preprocessing for topic modeling
        vectorizer = CountVectorizer(stop_words="english", max_features=5000)
        
        # Attempt to create the document-term matrix
        try:
            encoded_docs = vectorizer.fit_transform([pdf_text])
            
            # Check if encoded_docs has any features
            if encoded_docs.shape[1] == 0:
                st.error("The document contains only stop words or unrecognizable text. Please try another PDF.")
            else:
                # Display extracted text (optional)
                st.write("Extracted Text:", pdf_text[:1000])  # Show first 1000 characters for preview
                
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
                
                # Get and display the dominant topic in the document
                dominant_topic, topic_score = model.get_dominant_topic(encoded_docs)
                st.write(f"Dominant Topic for the Document: Topic {dominant_topic[0]} with Score: {topic_score[0]:.4f}")
        
        except ValueError as e:
            st.error(f"Error processing the document: {e}")
