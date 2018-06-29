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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

from timeline_features import make_labels, upsample_minority, make_feature_mat
from timeline_features import known_one_hot, similarity

np.set_printoptions(precision=4, suppress=True)

#------------------------------------------------------------------------------ 
#        IMPORT THE DATA!!
#------------------------------------------------------------------------------
filename = '../data/cb_input_datasets_full.pkl'
tf, df = pickle.load(open(filename, 'rb'))

# TODAY = '2013/12/12' # date of snapshot YYYY/MM/DD
TODAY = tf.dates.max() # most recent date in timeline

# Label data
# Y, threshold = make_labels(tf, df)
# pickle.dump([Y, threshold], open('../data/y_labels_median_0std.pkl', 'wb'))
Y, threshold = pickle.load(open('../data/y_labels_median_0std.pkl', 'rb'))

# Get unlabeled companies ("young companies")
unlabeled_ids = Y.loc[Y.label == 3].id
labeled_ids = Y.loc[~Y.id.isin(unlabeled_ids)].id
ages = tf.loc[tf.id.isin(unlabeled_ids), ['id', 'time_to_event']]\
         .groupby('id').max().time_to_event 

y = Y.loc[Y.id.isin(labeled_ids)]

# Initialize pointers
pred = pd.DataFrame()
fm = pd.Series()
sim_idx = {}

clf = RandomForestClassifier(criterion='entropy', max_depth=None, 
                             max_features=4, min_samples_split=3, n_jobs=10)

# Build dictionary of feature matrices. One per unlabeled company
tf_lab = tf.loc[tf.id.isin(labeled_ids)]
df_lab = df.loc[df.id.isin(labeled_ids)]

count = 0
MAX_COUNT = 1200
n_neighbors = 5

for _, ul in unlabeled_ids.iteritems():
    #-------------------------------------------------------------------------- 
    #        Build Features
    #--------------------------------------------------------------------------
    print('{:3d} Building features for {}'.format(count, ul))
    # Augment dataframes to include one unlabeled company
    tf_aug = tf_lab.append(tf[tf.id == ul])
    df_aug = df_lab.append(df[df.id == ul])

    # Cut off by company age!!
    tf_aug = tf_aug.loc[tf_aug.time_to_event < ages[ul]]

    # Make matrix of features
    X_aug = make_feature_mat(tf_aug, df_aug) # shape (Nl+1, m)

    # Includes categorical values
    feat_cols = list(set(X_aug.columns) - set(['id']))

    #-------------------------------------------------------------------------- 
    #        Train and predict
    #--------------------------------------------------------------------------
    print('Training model...')
    # Split out unknown and get similarity of single vector
    s = X_aug.loc[X_aug.id == ul, feat_cols].copy() # (m, 1) features of unknown company
    X = X_aug.loc[X_aug.id != ul, feat_cols].copy() # (Nl, m) features of labeled companies

    # Cosine-similarity for unlabeled->labeled
    C = similarity(s, X) # (m, 1) similarity vector
    # To predict: use single nearest neighbor with cosine similarity
    idx = C.values.argsort(axis=0).squeeze()[::-1] # array shape (m,) descending
    sim_idx[ul] = df.loc[df.id == ul].index.append(C.iloc[idx[:n_neighbors]].index)

    # When we build the X matrix, the index is unaligned from y, so we need to
    # realign the values according to the id column
    X, y_up = upsample_minority(X, y, maj_lab=2)
    yb = known_one_hot(y_up)

    # Don't use "train", "test" here to avoid over-writing previous time break
    X_train, X_test, y_train, y_test = train_test_split(X, yb, train_size=0.6, 
                                                        stratify=yb,
                                                        random_state=56)

    # More advanced model: let the computer do the work
    clf.fit(X_train, y_train)

    # Store in output dataframes
    y_hat = clf.predict(X_test)
    fm_i = precision_recall_fscore_support(y_test, y_hat, average='macro')
    fm = fm.append(pd.Series({ul:fm_i}))

    # predict for the single unknown company!
    pred_i = clf.predict(s).squeeze() # just get single vector
    pred = pred.append(pd.DataFrame.from_dict({ul:pred_i}, orient='index'))

    count += 1
    if count == MAX_COUNT:
        break

# filename = "../data/timeline_output_rf{}.pkl".format(MAX_COUNT)
# print('Writing to file {}...'.format(filename))
# pickle.dump([pred, ages, unlabeled_ids, fm], open(filename, 'wb'))

#------------------------------------------------------------------------------ 
#        Create output file for Flask app
#------------------------------------------------------------------------------
tf_fund = tf.loc[tf.event_id == 'funded', ['id', 'dates', 'famt_cumsum', 'time_to_event']]
# Dummy out:
# sim_idx = {'c:516': pd.Int64Index([3177, 3373, 18874, 3630, 14942, 14504], dtype='int64')}
filename = '../data/flask_db.pkl'
print('Writing to file {}...'.format(filename))
pickle.dump([pred, sim_idx, df, tf_fund, y], open(filename, 'wb'))

print('done.')
#==============================================================================
#==============================================================================
