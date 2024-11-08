import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from model import Model
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text

# Streamlit application setup
st.title("Enhanced Topic Modeling from PDF Files with Visualizations")

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

                # Visualize topics using LDAvis
                if model_type == "lda":
                    st.write("LDAvis Visualization:")
                    lda_vis_data = pyLDAvis.sklearn.prepare(model.model, encoded_docs, vectorizer)
                    pyLDAvis_html = pyLDAvis.prepared_data_to_html(lda_vis_data)
                    st.components.v1.html(pyLDAvis_html, width=800, height=600)
                
                # Bar plot of the topic-word distribution
                st.write("Topic-Word Distribution:")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=topics_df, x="Weight", y="Term", hue="Topic", dodge=False, ax=ax
                ).set_title("Top Words in Topics")
                st.pyplot(fig)
                
                # Correlation matrix between topics
                st.write("Topic Correlation Matrix:")
                topic_distributions = model.get_topic_distribution(encoded_docs)
                topic_corr = pd.DataFrame(topic_distributions).corr()
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(topic_corr, annot=True, cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)
        
        except ValueError as e:
            st.error(f"Error processing the document: {e}")
