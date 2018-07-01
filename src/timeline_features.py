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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelBinarizer, label_binarize
from sklearn.utils import resample

feat_cols = []

classes = ['Timely Exit', 'Late Exit', 'Slow Growth']
class_labels = [0, 1, 2]

def make_labels(tf, df):
    """Label company outcomes given timeline and static dataframe."""
    y = df[['id']].copy()
    y['label'] = np.nan

    # What is a successful exit? 
    # "Exit" == acquisition or IPO before median age of
    # acquisition for a given industry
    exits = tf.loc[(tf.event_id == 'public') | (tf.event_id == 'acquired')]\
              .merge(df, on='id', how='inner')
    g = exits.groupby('category_code')
    threshold = g.median().time_to_event #+ g.std().time_to_event # for ALL acquisitions

    # Companies that have not exited
    others = tf.loc[~tf.id.isin(exits.id)]\
               .groupby('id', as_index=False)[['event_id', 'time_to_event']].max()\
               .merge(df, on='id', how='inner')

    # Set age threshold for each label
    exits['threshold'] = np.nan
    others['threshold'] = np.nan
    for label in threshold.index:
        exits.loc[exits.category_code == label, 'threshold'] = threshold[label]
        others.loc[others.category_code == label, 'threshold'] = threshold[label]

    # Companies that have exited before threshold
    timely_exits = exits.loc[exits.time_to_event < exits.threshold]
    y.loc[y.id.isin(timely_exits.id), 'label'] = 0

    # Companies that have exited, but beyond the threshold
    late_exits = exits.loc[exits.time_to_event >= exits.threshold]
    y.loc[y.id.isin(late_exits.id), 'label'] = 1

    # Companies who are beyond threshold, but haven't exited
    dinosaurs = others.loc[others.time_to_event >= others.threshold]
    y.loc[y.id.isin(dinosaurs.id), 'label'] = 2

    # Everything else is a "young company"
    y.fillna(value=3, inplace=True)

    return y, threshold

def make_feat_cols(X, tf, label):
    """Add mean time to event and number of events to features."""
    global feat_cols
    tt = tf.loc[tf.event_id == label, ['id', 'time_diff']].groupby('id').mean()
    col_name = 'mean_' + label + '_time'
    tt.rename(columns={'time_diff':col_name}, inplace=True)
    X = X.join(tt, on='id')
    feat_cols += [col_name]

    tt = tf.loc[tf.event_id == label, ['id', 'event_count']].groupby('id').max()
    col_name = label + '_events'
    tt.rename(columns={'event_count':col_name}, inplace=True)
    X = X.join(tt, on='id')
    feat_cols += [col_name]
    return X

def make_feature_mat(tf, df):
    """Make matrix of features."""
    global feat_cols
    # Init with static features
    feat_cols = ['latitude', 'longitude', 'offices', 'products', 'experience']
    X = df[['id'] + feat_cols].copy()

    tt = tf.loc[tf.event_id == 'funded', ['id', 'famt']].groupby('id').mean()
    col_name = 'mean_famt'
    tt.rename(columns={'famt':col_name}, inplace=True)
    X = X.join(tt, on='id')
    feat_cols += [col_name]

    tt = tf.loc[tf.event_id == 'funded', ['id', 'famt_cumsum']].groupby('id').max()
    col_name = 'famt_cumsum'
    tt.rename(columns={'famt':col_name}, inplace=True)
    X = X.join(tt, on='id')
    feat_cols += [col_name]

    feats = ['funded', 'milestone', 'investment', 'acquisition']
    for f in feats:
        X = make_feat_cols(X, tf, f)

    # Add category as feature columns
    dvs = pd.get_dummies(df.category_code)
    dvs['id'] = df['id']
    X = X.merge(dvs, on='id', left_index=True) # DO NOT reset index!
    cats = list(dvs.drop('id', axis=1).columns)

    # fill NaNs and normalize only the numerical features
    X.fillna(X[feat_cols].median(), inplace=True)
    X.fillna(value=0, inplace=True)
    X[feat_cols] = StandardScaler().fit_transform(X[feat_cols])
    feat_cols += cats
    return X

def similarity(x, F):
    """Build similarity vector between two entities."""
    # Convert to just numerical matrices
    xv = x.loc[:, x.columns != 'id'].values.reshape(1, -1)
    Fv = F.loc[:, F.columns != 'id']
    C = cosine_similarity(Fv, xv).squeeze()
    C = pd.DataFrame(data=C, index=F.index).abs() # column vector
    return C

def known_one_hot(y, unk_lab=3):
    """Convert vector of known labels to one hot."""
    y_lab = y[y.label != unk_lab].label

    # Binarize the labels
    lb = LabelBinarizer()
    yb = lb.fit_transform(y_lab)
    # yb = label_binarize(y_lab.label, classes=class_labels)
    y = pd.DataFrame(data=yb, index=y_lab.index)
    return y, lb

def upsample_minority(X_in, y_in, maj_lab=2, n_classes=3):
    """Create upsampled versions to balance classes."""
    # Upsample minority classes
    X_maj = X_in.loc[y_in.label == maj_lab] # majority (y_in.label.value_counts())
    y_maj = y_in.loc[y_in.label == maj_lab]
    X_min_u = []
    y_min_u = []
    for i in range(n_classes-1):
        # Upsample minority class
        X_min_u.append(resample(X_in.loc[y_in.label == i], 
                            replace=True, 
                            n_samples=X_maj.shape[0],
                            random_state=56))
        y_min_u.append(resample(y_in.loc[y_in.label == i], 
                            replace=True, 
                            n_samples=X_maj.shape[0],
                            random_state=56))

    # Combine majority class with upsampled minority class
    X = pd.concat([X_maj] + X_min_u)
    y = pd.concat([y_maj] + y_min_u)
    return X, y

#==============================================================================
#==============================================================================
