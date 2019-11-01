import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_tweets(tweets):
    """takes a tweet as input and returns features as a numpy array"""
    vectorizer = TfidfVectorizer()
    keys = tweets.keys()
    list_tweets = [tweets[key] for key in keys]
    X = vectorizer.fit_transform(list_tweets)
    text_ft = {}
    for i, key in enumerate(keys):
        text_ft[key] = np.array(X[i].todense())[0]
        # print(X[i].todense().shape)
    return text_ft
