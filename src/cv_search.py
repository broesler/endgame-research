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

from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report

from timeline_features import upsample_minority

feat_cols = ['experience', 'latitude', 'longitude', 'offices', 'products', 'mean_fund_time',
             'funding_rounds', 'mean_funding_amt', 'cumulative_famt',
             'mean_milestone_time', 'milestones', 'mean_investment_time',
             'investments', 'mean_acquisition_time', 'acquisitions']

#------------------------------------------------------------------------------ 
#       Perform gridsearch on most dense data
#------------------------------------------------------------------------------
in_file = '../data/train_inputs_N5.pkl'
X_test, y_test, unlabeled_ids, clf = pickle.load(open(in_file, 'rb'))

out_file = '../data/timeline_output_test_svc_N5.pkl'
pred, ages, score, f1, fm = pickle.load(open(out_file, 'rb'))

# Get feature matrix for max age
key_age_max = max(ages, key=(lambda key: ages[key]))
X_max = X_test[key_age_max]
X_max = X_max.loc[~X_max.id.isin(unlabeled_ids)]
y_max = y_test.loc[~y_test.id.isin(unlabeled_ids)]

X, y = upsample_minority(X_max[feat_cols], y_max[['label']], maj_lab=2)
# pickle.dump([X_max, y_max, X, y], open('../data/Xy_age_5yrs.pkl', 'wb'))

# Binarize the labels
lb = LabelBinarizer()
yb = lb.fit_transform(y)
yb = pd.DataFrame(data=yb, index=y.index)

# Take 90% for training, rest is hold-out
X_train, X_test, y_train, y_test = train_test_split(X, yb, train_size=0.9, 
                                                    stratify=yb,
                                                    random_state=56)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# clf = KNeighborsClassifier(n_jobs=10)
# params = {'n_neighbors': np.arange(1, 50),
#           'weights': ['uniform', 'distance']}

# clf = SVC(kernel='rbf', C=100)
# params = {'C': scipy.stats.expon(scale=100), 
#           'gamma': scipy.stats.expon(scale=0.1)}
# y_train = lb.inverse_transform(y_train.values) # SVM needs normal values

clf = RandomForestClassifier(n_jobs=-1)
params = {'max_depth': [3, 5, None],
          'max_features': randint(1, 2*np.sqrt(X.shape[1])),
          'min_samples_split': randint(2, 51),
          'criterion': ['gini', 'entropy']}

# run randomized search
n_iter_search = 50
random_search = RandomizedSearchCV(clf, param_distributions=params,
                                   n_iter=n_iter_search, scoring='f1_macro',
                                   verbose=10, n_jobs=10)

# Run the test!
start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# Try Top-performing on test set
# Parameters: {'criterion': 'entropy', 'max_depth': None, 'max_features': 4, 'min_samples_split': 3}
# Parameters: {'C': 69.819976631697372, 'gamma': 0.5618553849000274}
# Parameters: {'weights': 'distance', 'n_neighbors': 2}

clf = KNeighborsClassifier(weights='distance', n_neighbors=2, n_jobs=10)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
#              precision    recall  f1-score   support
#
#           0       0.90      0.93      0.91       348
#           1       0.85      0.79      0.82       348
#           2       0.75      0.74      0.75       348
#
# avg / total       0.83      0.82      0.83      1044

# CHOOSE RFC: best precision on winners
clf = RandomForestClassifier(criterion='entropy', max_depth=None, 
                             max_features=4, min_samples_split=3)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
#              precision    recall  f1-score   support
#
#           0       0.94      0.91      0.92       348
#           1       0.88      0.78      0.83       348
#           2       0.73      0.70      0.72       348
#
# avg / total       0.85      0.80      0.82      1044

clf = SVC(C=69.82, gamma=0.562)
y_tr = lb.inverse_transform(y_train.values)
clf.fit(X_train, y_tr)
print(classification_report(lb.inverse_transform(y_test.values), clf.predict(X_test)))
#              precision    recall  f1-score   support
#
#         0.0       0.78      0.85      0.82       348
#         1.0       0.77      0.62      0.69       348
#         2.0       0.65      0.72      0.68       348
#
# avg / total       0.73      0.73      0.73      1044

#==============================================================================
#==============================================================================
