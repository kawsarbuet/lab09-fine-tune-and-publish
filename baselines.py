"""Classical baselines fitted only on the provided sample."""
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def fit_baselines(train, seed):
    models = {
        'majority': DummyClassifier(strategy='most_frequent'),
        'tfidf': make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
    }
    for model in models.values():
        model.fit(train.text, train.label)
    return models
