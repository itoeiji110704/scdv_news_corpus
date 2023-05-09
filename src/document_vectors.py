import re
from typing import List, Tuple

import numpy as np
from gensim.models import Word2Vec  # type: ignore
from gensim.models.doc2vec import Doc2Vec, TaggedDocument  # type: ignore
from sklearn import mixture
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from src.constants import (
    CONTEXT,
    DOWNSAMPLING,
    EPOCH_NUM,
    FEATURE_NUM,
    MIN_WORD_COUNT,
    STOP_WORDS,
)
from src.utils import timeit


def analyzer(text: str) -> List[str]:
    """Text analyzer that clean text and to be corpus.

    Args:
        text (str): text.

    Returns:
        List[str]: word list of the input text.
    """
    # Clean text and split to words
    text = text.lower()
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = re.sub(re.compile(r"[!-\/:-@[-`{-~]"), " ", text)
    text = text.split(" ")

    # Create word list of the sentence with pickup word maybe useful for analysis
    words = []
    for word in text:

        # Remove word that have number
        if re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None:
            continue

        # Remove stop word
        if word in STOP_WORDS:
            continue

        # Remove word that length is 0 or 1
        if len(word) < 2:
            continue

        words.append(word)

    return words


# Prepare functions to calculate document vectors using some methods.
# Prepare well known methods e.g. BoW, TF-IDF, averaged Word2Vec, Doc2Vec.


@timeit
def calc_bow(corpus: np.ndarray) -> np.ndarray:
    """Calculate BoW.

    Args:
        corpus (np.ndarray): corpus.

    Returns:
        np.ndarray: BoW document vector.
    """
    count_vectorizer = CountVectorizer(
        analyzer=analyzer, min_df=MIN_WORD_COUNT, binary=True
    )
    bows = count_vectorizer.fit_transform(corpus)
    return bows.toarray()


@timeit
def calc_tfidf(corpus: np.ndarray) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Calculate TF-IDF.

    Args:
        corpus (np.ndarray): corpus.

    Returns:
        Tuple[np.ndarray, TfidfVectorizer]: TF-IDF document vector. And to SCDV, return Tfidf vectorizer too.
    """
    tfidf_vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=MIN_WORD_COUNT)
    tfidfs = tfidf_vectorizer.fit_transform(corpus)
    return tfidfs.toarray(), tfidf_vectorizer


@timeit
def calc_avg_word2vec(corpus: np.ndarray) -> Tuple[np.ndarray, Word2Vec]:
    """Calculate averated word2vec.

    Args:
        corpus (np.ndarray): corpus.

    Returns:
        Tuple[np.ndarray, Word2Vec]: Averaged Word2Vec document vector. And to SCDV, return Word2Vec class too.
    """
    in_corpus = [analyzer(text) for text in corpus]
    word2vecs = Word2Vec(
        sentences=in_corpus,
        epochs=EPOCH_NUM,
        vector_size=FEATURE_NUM,
        min_count=MIN_WORD_COUNT,
        window=CONTEXT,
        sample=DOWNSAMPLING,
    )
    avg_word2vecs = np.array(
        [
            word2vecs.wv[list(analyzer(text) & word2vecs.wv.key_to_index.keys())].mean(
                axis=0
            )
            for text in corpus
        ]
    )
    return avg_word2vecs, word2vecs


@timeit
def calc_doc2vec(corpus: np.ndarray) -> np.ndarray:
    """Calculate Doc2Vec.

    Args:
        corpus (np.ndarray): corpus.

    Returns:
        np.ndarray: Doc2Vec document vector.
    """
    in_corpus = [
        TaggedDocument(words=analyzer(text), tags=[i]) for i, text in enumerate(corpus)
    ]
    doc2vecs = Doc2Vec(
        documents=in_corpus,
        dm=1,
        epochs=EPOCH_NUM,
        vector_size=FEATURE_NUM,
        min_count=MIN_WORD_COUNT,
        window=CONTEXT,
        sample=DOWNSAMPLING,
    )
    doc2vecs = np.array([doc2vecs.infer_vector(analyzer(text)) for text in corpus])
    return doc2vecs


# Prepare a function to calculate document vector using SCDV.
# SCDV is a method to calculate document vector proposed by Microsoft Research & IIT Kanpur.
# See https://arxiv.org/abs/1612.06778


@timeit
def calc_scdv(
    corpus: np.ndarray,
    word2vecs: Word2Vec,
    tfidf_vectorizer: TfidfVectorizer,
    clusters_num: int = 60,
    p: float = 0.04,
) -> np.ndarray:
    """Calculate SCDV. ref. https://arxiv.org/abs/1612.06778

    Args:
        corpus (np.ndarray): corpus.
        word2vecs (Word2Vec): Word2Vec class for this corpus.
        tfidf_vectorizer (TfidfVectorizer): TF-IDF vectorizer for this corpus.
        clusters_num (int, optional): the number of GMM cluster. Defaults to 60.
        p (float, optional): the threshold percentage for making it sparse. Defaults to 0.04.

    Returns:
        np.ndarray: SCDV document vector.
    """
    word_vectors = word2vecs.wv.vectors
    gmm = mixture.GaussianMixture(
        n_components=clusters_num, covariance_type="tied", max_iter=50
    )
    gmm.fit(word_vectors)

    idf_dic = dict(
        zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer._tfidf.idf_)
    )
    assign_dic = dict(zip(word2vecs.wv.index_to_key, gmm.predict(word_vectors)))
    soft_assign_dic = dict(
        zip(word2vecs.wv.index_to_key, gmm.predict_proba(word_vectors))
    )

    word_topic_vecs = {}
    for word in assign_dic:
        word_topic_vecs[word] = np.zeros(FEATURE_NUM * clusters_num, dtype=np.float32)
        for i in range(0, clusters_num):
            try:
                word_topic_vecs[word][i * FEATURE_NUM : (i + 1) * FEATURE_NUM] = (
                    word2vecs.wv[word] * soft_assign_dic[word][i] * idf_dic[word]
                )
            except:
                continue

    scdvs = np.zeros((len(corpus), clusters_num * FEATURE_NUM), dtype=np.float32)

    a_min = 0
    a_max = 0

    for i, text in enumerate(corpus):
        tmp = np.zeros(clusters_num * FEATURE_NUM, dtype=np.float32)
        words = analyzer(text)
        for word in words:
            if word in word_topic_vecs:
                tmp += word_topic_vecs[word]
        norm = np.sqrt(np.sum(tmp**2))
        if norm > 0:
            tmp /= norm
        a_min += min(tmp)
        a_max += max(tmp)
        scdvs[i] = tmp

    a_min = a_min * 1.0 / len(corpus)
    a_max = a_max * 1.0 / len(corpus)
    thres = (abs(a_min) + abs(a_max)) / 2
    thres *= p

    scdvs[abs(scdvs) < thres] = 0
    return scdvs
