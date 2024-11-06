from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from gensim import models
from gensim.corpora.dictionary import Dictionary
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

class Model:
    """Class for topic modeling with support for LDA, NMF, and optionally CTM."""
    
    def __init__(self,
                 num_topics: int,
                 docs: Union[pd.Series, List[List[str]]],
                 encoded_docs: Union[pd.Series, List[List[str]]],
                 filtered_lemmas: Union[pd.Series, List[List[str]]],
                 model_type: str = "lda",
                 random_state: int = 42,
                 **kwargs):
        self.model_type = model_type
        self.encoded_docs = encoded_docs
        self.num_topics = num_topics

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

    def get_topics(self, num_words: int = 10) -> Tuple[pd.DataFrame, bool]:
        if self.model_type == "lda" or self.model_type == "nmf":
            feature_names = self.encoded_docs.get_feature_names_out()
            topic_words = []
            for idx, topic in enumerate(self.int_model.components_):
                topic_words.extend(
                    [[idx, feature_names[i], topic[i]] for i in topic.argsort()[:-num_words - 1:-1]]
                )
            return pd.DataFrame(topic_words, columns=["Topic", "Word", "Weight"]), False

    def get_topics_list(self, dictionary: Dictionary, num_words: int = 20) -> List[List[str]]:
        if self.model_type == "lda" or self.model_type == "nmf":
            feature_names = dictionary.get_feature_names_out()
            return [[feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
                    for topic in self.int_model.components_]

    def get_topic_probs(self, corpus: Union[pd.Series, List[List[str]]]) -> np.ndarray:
        return self.int_model.transform(corpus)

    def save(self, path):
        # Save model based on sklearn/gensim's save methods
        if self.model_type in ["lda", "nmf"]:
            from joblib import dump
            dump(self.int_model, path)
