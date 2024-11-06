from typing import List, Tuple
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class Model:
    """Class for topic modeling with support for LDA and NMF."""
    
    def __init__(self,
                 num_topics: int,
                 encoded_docs,
                 vectorizer: CountVectorizer,
                 model_type: str = "lda",
                 random_state: int = 42,
                 **kwargs):
        self.model_type = model_type
        self.encoded_docs = encoded_docs
        self.num_topics = num_topics
        self.vectorizer = vectorizer

        if self.model_type == "lda":
            self.int_model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=random_state,
                max_iter=kwargs.get("max_iter", 10)
            )
            self.int_model.fit(encoded_docs)
        
        elif self.model_type == "nmf":
            self.int_model = NMF(
                n_components=num_topics,
                random_state=random_state,
                max_iter=kwargs.get("max_iter", 10)
            )
            self.int_model.fit(encoded_docs)

    def get_topics(self, num_words: int = 10) -> pd.DataFrame:
        """Extracts topics with top words."""
        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = []
        for idx, topic in enumerate(self.int_model.components_):
            topic_words.extend(
                [[idx, feature_names[i], topic[i]] for i in topic.argsort()[:-num_words - 1:-1]]
            )
        return pd.DataFrame(topic_words, columns=["Topic", "Word", "Weight"])

    def get_topic_probs(self):
        """Get topic probabilities for each document."""
        return self.int_model.transform(self.encoded_docs)

    def save(self, path):
        """Save the model."""
        from joblib import dump
        dump(self.int_model, path)
