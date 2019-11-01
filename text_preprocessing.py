import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_tweets(tweets):
    """
    Args:
        tweets: dict[int -> text]
    Returns:
        text_tf: dict[int -> np.array]
    """

    vectorizer = TfidfVectorizer(max_features = 100)
    keys = tweets.keys()
    list_tweets = [tweets[key] for key in keys]
    X = vectorizer.fit_transform(list_tweets)
    text_ft = {}
    for i, key in enumerate(keys):
        text_ft[key] = np.array(X[i].todense())[0]
    return text_ft
