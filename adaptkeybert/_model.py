import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from typing import List, Union, Tuple
import torch
from tqdm import tqdm
import time
from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from adaptkeybert._mmr import mmr
from adaptkeybert._maxsum import max_sum_distance
from adaptkeybert._highlight import highlight_document
from adaptkeybert.backend._utils import select_backend
from adaptkeybert._attention import DomainAttentionLayer


class KeyBERT:
    """
    A minimal method for keyword extraction with BERT

    The keyword extraction is done by finding the sub-phrases in
    a document that are the most similar to the document itself.

    First, document embeddings are extracted with BERT to get a
    document-level representation. Then, word embeddings are extracted
    for N-gram words/phrases. Finally, we use cosine similarity to find the
    words/phrases that are the most similar to the document.

    The most similar words could then be identified as the words that
    best describe the entire document.
    """

    def __init__(self, model="all-MiniLM-L6-v2", domain_adapt=False, zero_adapt=False):
        """KeyBERT initialization

        Arguments:
            model: Use a custom embedding model.
                   The following backends are currently supported:
                      * SentenceTransformers
                      * 🤗 Transformers
                      * Flair
                      * Spacy
                      * Gensim
                      * USE (TF-Hub)
                    You can also pass in a string that points to one of the following
                    sentence-transformers models:
                      * https://www.sbert.net/docs/pretrained_models.html
        """
        self.model = select_backend(model)
        self.domain_adapt = domain_adapt
        self.zero_adapt = zero_adapt
        if self.domain_adapt:
            self.attention_layer = None
            self.target_keywords = []
        if self.zero_adapt:
            self.domain_words = None
            self.domain_word_embeddings = None
            self.adaptive_thr = 0.0

    def zeroshot_pre_train(
        self,
        domain_words: List[str],
        adaptive_thr: float = 0.15,
        minimal_similarity_zeroshot: float = 0.8,
        ):
        if not self.zero_adapt:
            raise Exception("Can be only pre-trained when zero_adapt is True!")
        self.domain_words = domain_words
        self.domain_word_embeddings = self.model.embed(self.domain_words)
        self.minimal_similarity_zeroshot = minimal_similarity_zeroshot
        self.adaptive_thr = adaptive_thr

    def apply_zero_adaptation(self, candidate_embeddings, doc_embedding):
        computed_embeddings = []
        for candidate_embedding in candidate_embeddings:
            candidate_embedding = candidate_embedding.reshape(1, -1)
            max_similarity = np.max(cosine_similarity(candidate_embedding, self.domain_word_embeddings))
            if max_similarity<self.minimal_similarity_zeroshot:
                computed_embeddings.append(candidate_embedding[0])
            else:
                temp_embedding = (1-self.adaptive_thr*max_similarity)*candidate_embedding+self.adaptive_thr*max_similarity*doc_embedding
                computed_embeddings.append(temp_embedding[0])
        computed_embeddings = np.stack(computed_embeddings)
        return computed_embeddings

    def pre_train(
        self,
        docs: List[str],
        train_candidates: List[List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
        epochs: int = 100,
        lr: float = 1e-4
        ):
        if not self.domain_adapt:
            raise Exception("Can be only pre-trained when domain_adapt is True!")
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        # Extract potential words using a vectorizer / tokenizer
        if vectorizer:
            count = vectorizer.fit(docs)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(docs)
            except ValueError:
                return []

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        df = count.transform(docs)

        # Extract embeddings
        doc_embeddings = self.model.embed(docs)
        word_embeddings = self.model.embed(words)
        target_word_list = []
        target_indices = []
        for train_candidate in train_candidates:
            target_indices_single = []
            for word in train_candidate:
                if word not in target_word_list:
                    target_word_list.append(word)
                target_indices_single.append(target_word_list.index(word))
            target_indices.append(target_indices_single)
        target_word_embeddings = self.model.embed(target_word_list)
        target_word_embeddings_pt = torch.from_numpy(target_word_embeddings)
        bar = tqdm(range(epochs))
        for epoch in bar:
            mse_target_running = 0.0
            mse_gen_running = 0.0
            for index, _ in enumerate(docs):
                try:
                    # Select embeddings
                    candidate_indices = df[index].nonzero()[1]
                    candidates = [words[i] for i in candidate_indices if words[i] not in train_candidates[index]]
                    candidate_embeddings = word_embeddings[candidate_indices]
                    doc_embedding = doc_embeddings[index].reshape(1, -1)
                    target_embedding = target_word_embeddings[target_indices[index]]
                    if self.domain_adapt:
                        if self.attention_layer is None:
                            self.attention_layer = DomainAttentionLayer(doc_embedding.shape[1])
                            self.optimizer = torch.optim.SGD(self.attention_layer.parameters(), lr=lr)
                            self.attention_layer.train()
                        candidate_embeddings_pt = torch.from_numpy(candidate_embeddings)
                        doc_embeddings_pt = torch.from_numpy(doc_embedding)
                        target_embedding_pt = torch.from_numpy(target_embedding)
                        self.attention_layer.zero_grad()
                        candidate_embeddings_ = self.attention_layer(candidate_embeddings_pt, target_word_embeddings_pt)
                        gen_dist_score = torch.tensordot(candidate_embeddings_, doc_embeddings_pt.transpose(0, 1), dims=1)
                        gen_og_dist_score = torch.tensordot(candidate_embeddings_pt, doc_embeddings_pt.transpose(0, 1), dims=1)
                        target_embedding_ = self.attention_layer(target_embedding_pt, target_word_embeddings_pt)
                        target_dist_score = torch.tensordot(target_embedding_, doc_embeddings_pt.transpose(0, 1), dims=1)
                        doc_target_dist_score = torch.tensordot(doc_embeddings_pt, doc_embeddings_pt.transpose(0, 1), dims=1)
                        mse_gen = torch.mean((gen_og_dist_score-gen_dist_score)**2)
                        mse_target = torch.mean((doc_target_dist_score-target_dist_score)**2)
                        loss = mse_gen + 2*mse_target
                        loss.backward()
                        self.optimizer.step()
                        mse_gen_running += mse_gen.item()
                        mse_target_running += mse_target.item()
                except ValueError:
                    raise Exception("Value Error")
            bar.set_description(str({"mse_target": round(mse_target_running/len(docs), 3), "mse_gen": round(mse_gen_running/len(docs), 3)}))
            time.sleep(1)
        bar.close()
        self.target_word_embeddings_pt = target_word_embeddings_pt

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        highlight: bool = False,
        seed_keywords: List[str] = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: The number of candidates to consider if `use_maxsum` is
                           set to True.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`
            highlight: Whether to print the document and highlight its keywords/keyphrases.
                       NOTE: This does not work if multiple documents are passed.
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keywords from a single document:

        ```python
        from keybert import KeyBERT

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(doc)
        ```

        To extract keywords from multiple documents,
        which is typically quite a bit faster:

        ```python
        from keybert import KeyBERT

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(docs)
        ```
        """
        # Check for a single, empty document
        if self.domain_adapt:
            if self.attention_layer is None:
                raise Exception("Domain Adaptation Layer must be pre-trained!")
                return []
            else:
                self.attention_layer.eval()

        if self.zero_adapt:
            if self.domain_words is None:
                raise Exception("Zero-Shot Domain Adaptation must be pre-trained with domain words!!")
                return []

        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        # Extract potential words using a vectorizer / tokenizer
        if vectorizer:
            count = vectorizer.fit(docs)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(docs)
            except ValueError:
                return []

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        df = count.transform(docs)

        # Extract embeddings
        doc_embeddings = self.model.embed(docs)
        word_embeddings = self.model.embed(words)

        # Find keywords
        all_keywords = []
        for index, _ in enumerate(docs):

            try:
                # Select embeddings
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)
                candidate_embeddings_pt = torch.from_numpy(candidate_embeddings)
                if self.domain_adapt:
                    candidate_embeddings_ = self.attention_layer(candidate_embeddings_pt, self.target_word_embeddings_pt).detach().numpy()
                    candidate_embeddings = np.average([candidate_embeddings, candidate_embeddings_], axis=0, weights=[2, 1])
                if self.zero_adapt:
                    candidate_embeddings = self.apply_zero_adaptation(candidate_embeddings, doc_embedding)

                # Guided KeyBERT with seed keywords
                if seed_keywords is not None:
                    seed_embeddings = self.model.embed([" ".join(seed_keywords)])
                    doc_embedding = np.average(
                        [doc_embedding, seed_embeddings], axis=0, weights=[3, 1]
                    )

                # Maximal Marginal Relevance (MMR)
                if use_mmr:
                    keywords = mmr(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        diversity,
                    )

                # Max Sum Distance
                elif use_maxsum:
                    keywords = max_sum_distance(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        nr_candidates,
                    )

                # Cosine-based keyword extraction
                else:
                    distances = cosine_similarity(doc_embedding, candidate_embeddings)
                    keywords = [
                        (candidates[index], round(float(distances[0][index]), 4))
                        for index in distances.argsort()[0][-top_n:]
                    ][::-1]

                all_keywords.append(keywords)

            # Capturing empty keywords
            except ValueError:
                all_keywords.append([])

        # Highlight keywords in the document
        if len(all_keywords) == 1:
            if highlight:
                highlight_document(docs[0], all_keywords[0], count)
            all_keywords = all_keywords[0]

        return all_keywords
