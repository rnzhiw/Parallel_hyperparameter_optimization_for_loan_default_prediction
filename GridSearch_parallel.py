# coding=utf-8
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# -------------download data
news = fetch_20newsgroups(subset='all')
# -------------select front 3000 data,split data，25% for test
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25,
                                                    random_state=33)
# -------------use Pipeline to simplify processing flow,contact wordsVectorizer with classifier model
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])
# -------------create geometric progression （等比数列），total 4*3 ＝12 parameters combination
parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}
# -------------n_jobs=-1 means use all available CPU
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)

time_ = gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_

print ('Best accuracy is:', gs.score(X_test, y_test))