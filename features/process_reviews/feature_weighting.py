
from sklearn.feature_extraction.text import TfidfVectorizer



def tfidf_bow(dataset):
    """
    Args:
        dataset: a collection of documents stored in a vector
    Returns:
        A list of words, corresponding to the indexed vocabulary of the dataset
    """
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(dataset)

    return x, vectorizer



def ngrams_tfidf(dataset, range_begin, range_end):
    """
    Args:
        dataset: a collection of documents stored in a vector
    Returns:
        A list of ngrams, corresponding to the indexed vocabulary of the dataset
    """
    ngram_vectorizer = TfidfVectorizer(ngram_range=(range_begin, range_end), token_pattern=r'\b\w+\b', min_df=1)
    x = ngram_vectorizer.fit_transform(dataset)

    return x, ngram_vectorizer


    return x, ngram_vectorizer

