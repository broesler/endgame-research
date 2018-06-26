#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: build_features.py
#  Created: 06/24/2018, 18:05
#   Author: Bernie Roesler
#
"""
  Description: Auxiliary functions to build features from given time slice
"""
#==============================================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

feat_cols = ['latitude', 'longitude', 'offices', 'products',
             'mean_fund_time', 'funding_rounds']

def make_labels(tf, df):
    """Label company outcomes given timeline and static dataframe."""
    y = df[['id']].copy()
    y['label'] = np.nan

    # Failures
    closures = tf.loc[tf.event_id == 'closed']
    y.loc[y.id.isin(closures.id), 'label'] = 0

    # What is a successful exit? 
    # "Exit" == acquisition or IPO at least before 1-std beyond mean age of
    # acquisition for a given industry
    # NOTE workaround: groupby().std() does NOT work on timeseries64, but
    # describe() does!?
    exits = tf.loc[(tf.event_id == 'public') | (tf.event_id == 'acquired')]\
              .merge(df, on='id', how='inner')
    g = exits.groupby('category_code')
    threshold = g.mean().time_to_event + g.std().time_to_event # for ALL acquisitions

    # Slow growth companies
    others = tf.loc[~(tf.id.isin(exits.id) | tf.id.isin(closures.id))]\
               .groupby('id', as_index=False).time_to_event.max()\
               .merge(df, on='id', how='inner')

    # Set age threshold for each label
    exits['threshold'] = np.nan
    for label in threshold.index:
        exits.loc[exits.category_code == label, 'threshold'] = threshold[label]
        others.loc[others.category_code == label, 'threshold'] = threshold[label]

    # Companies that have exited in a timely manner
    timely_exits = exits.loc[exits.time_to_event < exits.threshold]
    y.loc[y.id.isin(timely_exits.id), 'label'] = 1

    # Companies that have exited, but beyond the threshold
    slow_exits = exits.loc[exits.time_to_event >= exits.threshold]
    y.loc[y.id.isin(slow_exits.id), 'label'] = 2

    # Companies who are beyond threshold, but haven't exited or closed
    dinosaurs = others.loc[others.time_to_event >= others.threshold]
    y.loc[y.id.isin(dinosaurs.id), 'label'] = 3

    # Everything else is 'unknown'
    y.fillna(value=4, inplace=True)

    return y

def make_features_dict(tf, df, y):
    """Build dictionary of feature matrices. One per unlabeled company."""
    labeled_ids = y.loc[y.label != 4].id
    tf_lab = tf.loc[tf.id.isin(labeled_ids)]
    df_lab = df.loc[df.id.isin(labeled_ids)]
    unlabeled_ids = y.loc[y.label == 4].id
    F = {}
    count = 0
    MAX_COUNT = 100
    for i, ul in unlabeled_ids.iteritems():
        # Augment dataframes to include one unlabeled company
        tf_aug = tf_lab.append(tf[tf.id == ul])
        df_aug = df_lab.append(df[df.id == ul])
        # Cut off by company age!!
        ul_age = tf.loc[tf.id == ul].time_to_event.max() 
        tf_aug = tf_aug.loc[tf_aug.time_to_event < ul_age]
        # Make matrix of features
        F[ul] = make_feature_mat(tf_aug, df_aug) # shape (Nl+1, m)
        count += 1
        if count > MAX_COUNT:
            break
    return F

def make_feature_mat(tf, df):
    """Make matrix of features."""
    # m = 10 # features
    # n = df.id.unique().shape[0]
    # X = pd.DataFrame(data=np.random.randn(n, 10))
    # X['id'] = df.id.unique()

    X = df.copy()

    tt = tf.loc[tf.event_id == 'funded'].groupby('id').mean().time_diff
    tt.name = 'mean_fund_time'
    X = X.join(tt, on='id')

    tt = tf.loc[tf.event_id == 'funded'].groupby('id').max().event_count
    tt.name = 'funding_rounds'
    X = X.join(tt, on='id')

    # Fill NaN values with 0s (funding event does not exist)
    X.fillna(value=0, inplace=True)

    # TODO add category as feature columns

    Xn = normalize(X[feat_cols])
    # Include non-normalized data again
    X = pd.concat([X.loc[:, ~X.columns.isin(feat_cols)], Xn], axis=1)
    return X

def normalize(X):
    X = X.fillna(X.median())
    Xn = StandardScaler().fit_transform(X)
    Xn = pd.DataFrame(data=Xn, columns=X.columns, index=X.index)
    return Xn

def similarity(x, F):
    """Build similarity vector between two entities."""
    # Convert to just numerical matrices
    xv = x.loc[:, x.columns != 'id'].values.reshape(1, -1)
    Fv = F.loc[:, F.columns != 'id']
    C = cosine_similarity(Fv, xv).squeeze()
    C = pd.DataFrame(data=C, index=F.index).abs() # column vector
    return C
#==============================================================================
#==============================================================================
