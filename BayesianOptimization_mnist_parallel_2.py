# coding=utf-8
import os
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
print(start)

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import tensorflow as tf
from tensorflow import keras


def load_mnist():
    path = r'data/mnist.npz'  # 放置mnist.py的目录。注意斜杠
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

# mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = load_mnist()

print(X_train.shape)

print(X_test.shape)

print(X_train.dtype)

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(y_train[index], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
save_fig("mnist_plot")
plt.show()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [{'bootstrap': [True],
     'max_depth': [6, 10],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [3, 5],
     'min_samples_split': [4, 6],
     'n_estimators': [100, 350]}
     ]

forest_clf = RandomForestClassifier()

# forest_grid_search = GridSearchCV(forest_clf, param_grid, cv=3,
#                                   scoring="accuracy",
#                                   return_train_score=True,
#                                   verbose=True,
#                                   n_jobs=-1)

# forest_grid_search.fit(X_train, y_train)
#
# print(forest_grid_search.best_params_)
#
# print(forest_grid_search.best_estimator_)
#
# print(forest_grid_search.best_score_)

# end = time.time()
# print(end)
# print('Running time: %s Seconds'%(end-start))

from sklearn.model_selection import RandomizedSearchCV

start2 = time.time()
param_space = {"bootstrap": [True],
        "max_depth": [6, 8, 10, 12, 14],
        "max_features": ['auto', 'sqrt','log2'],
        "min_samples_leaf": [2, 3, 4],
        "min_samples_split": [2, 3, 4, 5],
        "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

# forest_rand_search = RandomizedSearchCV(forest_clf, param_space, n_iter=32,
#                                         scoring="accuracy", verbose=True, cv=3,
#                                         n_jobs=-1, random_state=42)

# forest_rand_search.fit(X_train, y_train)
#
# print(forest_rand_search.best_params_)
#
# print(forest_rand_search.best_estimator_)
#
# print(forest_rand_search.best_score_)

# end2 = time.time()
# print(end2)
# print('Running time: %s Seconds'%(end2-start2))

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

start3 = time.time()

search_space = {"bootstrap": Categorical([True, False]), # values for boostrap can be either True or False
        "max_depth": Integer(6, 20), # values of max_depth are integers from 6 to 20
        "max_features": Categorical(['auto', 'sqrt','log2']),
        "min_samples_leaf": Integer(2, 10),
        "min_samples_split": Integer(2, 10),
        "n_estimators": Integer(100, 500)
    }

def on_step(optim_result):
    """
    Callback meant to view scores after
    each iteration while performing Bayesian
    Optimization in Skopt"""
    score = forest_bayes_search.best_score_
    score = forest_bayes_search.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

forest_bayes_search = BayesSearchCV(forest_clf, search_space, n_iter=32, # specify how many iterations
                                    scoring="accuracy", n_jobs=2, cv=3)

forest_bayes_search.fit(X_train, y_train, callback=on_step)

print(forest_bayes_search.best_params_)

print(forest_bayes_search.best_estimator_)

print(forest_bayes_search.best_score_)

end = time.time()
print(end)
print('Running time: %s Seconds'%(end-start))