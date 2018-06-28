#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: train_model.py
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
tf = tf.sort_values('dates')

# Get corresponding static data
df = df[df.id.isin(tf.id)]

# Label data
y = make_labels(tf, df)
# pickle.dump(y, open('../data/y_labels_median_0std.pkl', 'wb'))

print('Building feature matrices...')
X, unlabeled_ids, ages = make_features_dict(tf, df, y)
# pickle.dump([X, y, unlabeled_ids, clf], open('../data/train_inputs_N5.pkl', 'wb'))

# n_neighbors = 2
# clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=10)
clf = RandomForestClassifier(criterion='entropy', max_depth=None, 
                             max_features=6, min_samples_split=4)
# clf = SVC(kernel='rbf', C=47.352, gamma=0.238)
print('Training...')
pred, score, fm, f1 = train_and_predict(X, y, unlabeled_ids, clf)

# filename = '../data/timeline_output_full_knn{}.pkl'.format(n_neighbors)
# filename = '../data/timeline_output_svc_N5.pkl'
# filename = '../data/timeline_output_rF_N5.pkl'
# print('Writing to file {}...'.format(filename))
# pickle.dump([pred, ages, score, f1, fm], open(filename, 'wb'))

print('done.')
#==============================================================================
#==============================================================================
