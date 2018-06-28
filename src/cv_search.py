#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: cv_search.py
#  Created: 06/28/2018, 02:51
#   Author: Bernie Roesler
#
"""
  Description: Cross-validation parameter search
"""
#==============================================================================

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from time import time
import seaborn as sns

from scipy.stats import randint

from sklearn.preprocessing import label_binarize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from timeline_features import feat_cols, known_one_hot, upsample_minority

plt.close('all')
sns.set_style('whitegrid')
# sns.set_context('poster', font_scale=1.6)
sns.set_context()

#------------------------------------------------------------------------------ 
#       Perform gridsearch on most dense data
#------------------------------------------------------------------------------
in_file = '../data/train_inputs_N5.pkl'
X_test, y_test, unlabeled_ids_test, clf = pickle.load(open(in_file, 'rb'))

# in_file = '../data/timeline_output_test_full_svc.pkl'
out_file = '../data/timeline_output_test_svc_N5.pkl'
pred_test, ages_test, score_test, f1_test, fm_test = pickle.load(open(out_file, 'rb'))

# Get feature matrix for max age
key_age_max = max(ages_test, key=(lambda key: ages_test[key]))
X_max = X_test[key_age_max]
X_max = X_max.loc[~X_max.id.isin(unlabeled_ids_test)]
y_max = y_test.loc[~y_test.id.isin(unlabeled_ids_test)]

# #------------------------------------------------------------------------------ 
# #        Pair plot of features
# #------------------------------------------------------------------------------
# df_pairs = X_max.merge(y_max, on='id', how='inner')
# for i in range(0, len(feat_cols), 5):
#     g = sns.pairplot(data=df_pairs,
#                     vars=feat_cols[i:i+5],
#                     hue='label')
# plt.show()

X, y = upsample_minority(X_max[feat_cols], y_max, maj_lab=2)

# Utility function to report best scores
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")

# clf = KNeighborsClassifier(n_jobs=10)
# params = {'n_neighbors': np.arange(1, 50),
#           'weights': ['uniform', 'distance']}

# clf = SVC(kernel='rbf', C=100)
# params = {'C': scipy.stats.expon(scale=100), 
#           'gamma': scipy.stats.expon(scale=0.1)}

# clf = RandomForestClassifier(n_jobs=10)
# params = {'max_depth': [3, None],
#           'max_features': randint(1, 2*np.sqrt(X.shape[1])),
#           'min_samples_split': randint(2, 51),
#           'criterion': ['gini', 'entropy']}

# run randomized search
# n_iter_search = 100
# random_search = RandomizedSearchCV(clf, param_distributions=params,
#                                    n_iter=n_iter_search, scoring='f1_macro',
#                                    verbose=10, n_jobs=10)
#
# start = time()
# random_search.fit(X, y.label)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." % ((time() - start), n_iter_search))
# report(random_search.cv_results_)

#------------------------------------------------------------------------------ 
#        Plot decision boundary
#------------------------------------------------------------------------------
n_neighbors = 2
clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
clf = RandomForestClassifier(criterion='entropy', max_depth=None, 
                             max_features=6, min_samples_split=4)

# Visualize result after dimensionality reduction using truncated SVD
svd = TruncatedSVD(n_components=2)
X_reduced = svd.fit_transform(X)

# scatter plot of original and reduced data
fig = plt.figure(figsize=(9, 8))

ax = plt.subplot(111)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y.label, s=50, edgecolor='k')
ax.set_title("Truncated SVD reduction (2D) of data ({:d}D)".format(X.shape[1]))

plt.show()

#------------------------------------------------------------------------------ 
#        Confusion matrix
#------------------------------------------------------------------------------
# p = pred.copy()
# p['id'] = p.index
# p.reset_index(drop=True, inplace=True)
#
# def inverse_binarize(yb):
#     """Inverse binarize."""
#     y = yb[[0]].copy()
#     y.loc[yb[0] == 1] = 0
#     y.loc[yb[1] == 1] = 1
#     y.loc[yb[2] == 1] = 2
#     y.loc[yb[3] == 1] = 3
#     return y
# confusion_matrix(y.loc[y.id.isin(pred.index), 'label'], inverse_binarize(pred))

#==============================================================================
#==============================================================================
