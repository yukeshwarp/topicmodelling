from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer


class Model:
    def __init__(self, num_topics: int, docs: List[str], filtered_lemmas: List[List[str]], model_type: str = "lda", random_state: int = 42, **kwargs):
        self.model_type = model_type
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
        self.corpus_matrix = self.vectorizer.fit_transform(filtered_lemmas)
        
        if self.model_type == "lda":
            self.int_model = LatentDirichletAllocation(
                n_components=num_topics, random_state=random_state, **kwargs
            )
            self.int_model.fit(self.corpus_matrix)
        elif self.model_type == "nmf":
            self.int_model = NMF(n_components=num_topics, random_state=random_state, **kwargs)
            self.int_model.fit(self.corpus_matrix)
        elif self.model_type == "ctm":
            self.tp = TopicModelDataPreparation(kwargs.get("contextualized_model", "paraphrase-distilroberta-base-v2"))
            self.training_dataset = self.tp.fit(
                text_for_contextual=docs,
                text_for_bow=[" ".join(text) for text in filtered_lemmas]
            )
            self.int_model = CombinedTM(bow_size=len(self.tp.vocab), contextual_size=kwargs.get("contextual_size", 768), n_components=num_topics)
            self.int_model.fit(self.training_dataset)

    def get_topics(self, num_words: int = 10) -> pd.DataFrame:
        if self.model_type in ["lda", "nmf"]:
            topic_words = []
            for topic_idx, topic in enumerate(self.int_model.components_):
                words = [(self.vectorizer.get_feature_names_out()[i], topic[i]) for i in topic.argsort()[:-num_words - 1:-1]]
                topic_words.append([topic_idx, words])
            return pd.DataFrame(topic_words, columns=["Topic", "Words"])
        elif self.model_type == "ctm":
            topic_words_distr = self.int_model.get_topic_word_distribution()
            words_ids = np.apply_along_axis(lambda x: x.argsort()[::-1][:num_words], 1, topic_words_distr)
            return pd.DataFrame([
                [topic_id, [self.tp.vocab[word_id] for word_id in word_ids]] for topic_id, word_ids in enumerate(words_ids)
            ], columns=["Topic", "Words"])
