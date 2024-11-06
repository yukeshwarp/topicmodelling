from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

class Model:
    """Class for topic modeling using LDA or NMF."""
    def __init__(self, num_topics, encoded_docs, vectorizer, model_type="lda", random_state=42):
        self.num_topics = num_topics
        self.model_type = model_type
        self.vectorizer = vectorizer

        if model_type == "lda":
            self.model = LatentDirichletAllocation(n_components=num_topics, random_state=random_state)
        elif model_type == "nmf":
            self.model = NMF(n_components=num_topics, random_state=random_state)
        
        # Fit the model to the document-term matrix
        self.model.fit(encoded_docs)

    def get_topics(self, num_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(self.model.components_):
            topic_terms = [(feature_names[i], topic[i]) for i in topic.argsort()[:-num_words - 1:-1]]
            topics.append((topic_idx, topic_terms))

        topics_df = pd.DataFrame(
            [[topic_idx, term, weight] for topic_idx, terms in topics for term, weight in terms],
            columns=["Topic", "Term", "Weight"]
        )
        return topics_df

    def get_topic_distribution(self, encoded_docs):
        # Get the topic distribution for each document
        return self.model.transform(encoded_docs)

    def get_dominant_topic(self, encoded_docs):
        topic_distribution = self.get_topic_distribution(encoded_docs)
        dominant_topics = np.argmax(topic_distribution, axis=1)
        dominant_scores = topic_distribution[np.arange(len(dominant_topics)), dominant_topics]
        return dominant_topics, dominant_scores
