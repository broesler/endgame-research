#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: test_model.py
#  Created: 06/13/2018, 15:11
#   Author: Bernie Roesler
#
"""
  Description: Create, train, and cross-validate initial model
"""
#==============================================================================

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from timeline_features import make_labels, make_features_dict, train_and_predict

np.set_printoptions(precision=4, suppress=True)

#------------------------------------------------------------------------------ 
#        IMPORT THE DATA!!
#------------------------------------------------------------------------------
# filename = '../data/cb_input_datasets.pkl' # just funding rounds
filename = '../data/cb_input_datasets_full.pkl'
tf, df = pickle.load(open(filename, 'rb'))

#------------------------------------------------------------------------------ 
#        Train/Test Split Timeseries
#------------------------------------------------------------------------------
# TODAY = '2013/12/12' # date of snapshot YYYY/MM/DD
TODAY = tf.dates.max() # most recent date in timeline

# mean time threshold to acquisition (all industries) is 12 years
TEST_WINDOW = pd.to_timedelta(6, unit='y')
TRAIN_END = TODAY - TEST_WINDOW

# Split values in time
X = tf.sort_values('dates')

# NOTE train using limited dataset, then test using rest of data (which will
# give future of companies founded within `threshold` years of the cutoff), but
# caveat to typical TimeSeriesSplit: we need to retain ALL of the history for
# the test set to properly build the features (not just the chunk of time
# between TRAIN_END and TODAY)
tf_train = X.loc[X.dates < TRAIN_END]
tf_test = X

# Get corresponding static data
df_train = df[df.id.isin(tf_train.id)]
df_test = df[df.id.isin(tf_test.id)]

# Label data
y_train = make_labels(tf_train, df_train)
y_test = make_labels(tf_test, df_test)
# pickle.dump(y_test, open('../data/y_labels_median_0std.pkl', 'wb'))

# TODO Calculate score on CHANGED values between train and test set unknowns
# TODO ONLY have to run predictions on CHANGED unknowns!!
# Instead of returning unlabeled ids, we can just provide a list over which to
# loop (should be ~5,000)
unlab_train = y_train.loc[y_train.label == 3]
unlab_test = y_test.loc[y_test.id.isin(unlab_train.id)]
unlab_diff = sum((unlab_train.label - unlab_test.label) > 0)

# Based on labels, need to build features comparing EVERY un-labeled company
# with every labeled company (i.e. cosine similarity), BUT, we need to cut off
# the `time_to_event` threshold for each comparison. We are building a matrix:
#   X : (n, m) for n unknown companies and m labeled companies
# where the (i, j)th entry is the similarity between the ith unknown company
# and the jth labeled company.

print('Building feature matrices...')
# X_train, unlabeled_ids_train, ages_train = make_features_dict(tf_train, df_train, y_train)
X_test, unlabeled_ids_test, ages_test = make_features_dict(tf_test, df_test, y_test)
# pickle.dump([X_test, y_test, unlabeled_ids_test, clf], open('../data/train_inputs_N5.pkl', 'wb'))

# n_neighbors = 5
# clf = KNeighborsClassifier(n_neighbors=n_neighbors)
# clf = OneVsRestClassifier(SVC(kernel='rbf', C=1)) # gamma='auto' --> 1/n_features
clf = SVC(kernel='rbf', C=100) # gamma='auto' --> 1/n_features
print('Training...')
# pred_train, score_train = train_and_predict(X_train, y_train, unlabeled_ids_train, clf)
pred_test, score_test, fm_test, f1_test = train_and_predict(X_test, y_test, unlabeled_ids_test, clf)

# filename = '../data/timeline_output_test_full_knn{}.pkl'.format(n_neighbors)
# filename = '../data/timeline_output_test_svc_N5.pkl'
# print('Writing to file {}...'.format(filename))
# pickle.dump([pred_test, ages_test, score_test, f1_test, fm_test], open(filename, 'wb'))


# Confusion matrix
# p_test = pred_test.copy()
# p_test['id'] = p_test.index
# p_test.reset_index(drop=True, inplace=True)
#
# def inverse_binarize(yb):
#     """Inverse binarize."""
#     y = yb[[0]].copy()
#     y.loc[yb[0] == 1] = 0
#     y.loc[yb[1] == 1] = 1
#     y.loc[yb[2] == 1] = 2
#     y.loc[yb[3] == 1] = 3
#     return y
# confusion_matrix(y_test.loc[y_test.id.isin(pred_test.index), 'label'], inverse_binarize(pred_test))

#------------------------------------------------------------------------------ 
#        Key outputs
#------------------------------------------------------------------------------
# >>> unlab = y_train.loc[y_train.label == 4]
# >>> unlab_test = y_test.loc[y_test.id.isin(unlab.id)]
# >>> unlab.label.value_counts()
# ===
# 4.0    10184
# Name: label, dtype: int64
# >>> unlab_test.label.value_counts()
# ===
# 4.0    4569
# 3.0    3709
# 0.0     697
# 2.0     683
# 1.0     526
# Name: label, dtype: int64
# >>> sum((unlab.label - unlab_test.label) > 0)
# === 5615

print('done.')
#==============================================================================
#==============================================================================
