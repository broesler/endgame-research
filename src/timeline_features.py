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

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.utils import resample

feat_cols = ['latitude', 'longitude', 'offices', 'products',
             'mean_fund_time', 'funding_rounds']

classes = ['Failed', 'Timely Exit', 'Late Exit', 'Slow Growth']

MAX_COUNT = 100

def make_labels(tf, df):
    """Label company outcomes given timeline and static dataframe."""
    y = df[['id']].copy()
    y['label'] = np.nan

    # What is a successful exit? 
    # "Exit" == acquisition or IPO before median age of
    # acquisition for a given industry
    # NOTE workaround: groupby().std() does NOT work on timeseries64, but
    # describe() does!?
    exits = tf.loc[(tf.event_id == 'public') | (tf.event_id == 'acquired')]\
              .merge(df, on='id', how='inner')
    g = exits.groupby('category_code')
    threshold = g.median().time_to_event #+ g.std().time_to_event # for ALL acquisitions

    # Companies that have not exited
    others = tf.loc[~tf.id.isin(exits.id)]\
               .groupby('id', as_index=False).time_to_event.max()\
               .merge(df, on='id', how='inner')

    # Set age threshold for each label
    exits['threshold'] = np.nan
    others['threshold'] = np.nan
    for label in threshold.index:
        exits.loc[exits.category_code == label, 'threshold'] = threshold[label]
        others.loc[others.category_code == label, 'threshold'] = threshold[label]

    # Failures -- close before threshold
    closures = others.loc[(others.event_id == 'closed') 
                          & (others.time_to_event < others.threshold)]
    y.loc[y.id.isin(closures.id), 'label'] = 0

    # Companies that have exited before threshold
    timely_exits = exits.loc[exits.time_to_event < exits.threshold]
    y.loc[y.id.isin(timely_exits.id), 'label'] = 1

    # Companies that have exited, but beyond the threshold
    late_exits = exits.loc[exits.time_to_event >= exits.threshold]
    y.loc[y.id.isin(late_exits.id), 'label'] = 2

    # Companies who are beyond threshold, but haven't exited
    # dinosaurs = others.loc[(others.event_id != 'closed') 
    #                       & (others.time_to_event >= others.threshold)]
    dinosaurs = others.loc[others.time_to_event >= others.threshold]
    y.loc[y.id.isin(dinosaurs.id), 'label'] = 3

    # Everything else is a "young company"
    y.fillna(value=4, inplace=True)

    return y

def make_features_dict(tf, df, y):
    """Build dictionary of feature matrices. One per unlabeled company."""
    labeled_ids = y.loc[y.label != 4].id
    tf_lab = tf.loc[tf.id.isin(labeled_ids)]
    df_lab = df.loc[df.id.isin(labeled_ids)]
    unlabeled_ids = y.loc[y.label == 4].id
    X = {}
    age = {}
    count = 0
    MAX_COUNT = 100
    for _, ul in unlabeled_ids.iteritems():
        print('Building features for {}'.format(ul))
        # Augment dataframes to include one unlabeled company
        tf_aug = tf_lab.append(tf[tf.id == ul])
        df_aug = df_lab.append(df[df.id == ul])
        # Cut off by company age!!
        age[ul] = tf.loc[tf.id == ul].time_to_event.max() 
        tf_aug = tf_aug.loc[tf_aug.time_to_event < age[ul]]
        # Make matrix of features
        X[ul] = make_feature_mat(tf_aug, df_aug) # shape (Nl+1, m)
        count += 1
        if count > MAX_COUNT:
            break
    return X, unlabeled_ids, age

def make_feature_mat(tf, df):
    """Make matrix of features."""
    # m = 10 # features
    # n = df.id.unique().shape[0]
    # X = pd.DataFrame(data=np.random.randn(n, 10))
    # X['id'] = df.id.unique()
    X = df.copy()

    # Funding Rounds
    tt = tf.loc[tf.event_id == 'funded'].groupby('id').mean().time_diff
    tt.name = 'mean_fund_time'
    X = X.join(tt, on='id')

    tt = tf.loc[tf.event_id == 'funded'].groupby('id').max().event_count
    tt.name = 'funding_rounds'
    X = X.join(tt, on='id')

    # Fill NaN values with 0s (funding event does not exist)
    X.fillna(value=0, inplace=True)

    # Milestones
    tt = tf.loc[tf.event_id == 'milestone'].groupby('id').mean().time_diff
    tt.name = 'mean_milestone_time'
    X = X.join(tt, on='id')

    tt = tf.loc[tf.event_id == 'milestone'].groupby('id').max().event_count
    tt.name = 'milestones'
    X = X.join(tt, on='id')

    # Investments
    tt = tf.loc[tf.event_id == 'investment'].groupby('id').mean().time_diff
    tt.name = 'mean_investment_time'
    X = X.join(tt, on='id')

    tt = tf.loc[tf.event_id == 'investment'].groupby('id').max().event_count
    tt.name = 'investments'
    X = X.join(tt, on='id')

    # Acquisitions
    tt = tf.loc[tf.event_id == 'acquisition'].groupby('id').mean().time_diff
    tt.name = 'mean_acquisition_time'
    X = X.join(tt, on='id')

    tt = tf.loc[tf.event_id == 'acquisition'].groupby('id').max().event_count
    tt.name = 'acquisitions'
    X = X.join(tt, on='id')

    # TODO add category as feature columns
    # TODO include # co-investors per round (subtract one before cumsum()

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

def known_one_hot(y):
    """Convert vector of known labels to one hot."""
    y_lab = y[y.label != 4]
    yb = label_binarize(y_lab.label, classes=[0, 1, 2, 3])
    y = pd.DataFrame(data=yb, index=y_lab.index)
    return y

def get_sorted_values(dx, dy):
    """Sort dicts by dx.items() and return arrays of dx and dy values."""
    sort_keys = [x[0] for x in sorted(dx.items(), key=lambda kv: kv[1])]
    x = np.array([dx[k] for k in sort_keys])
    y = np.array([dy[k] for k in sort_keys])
    return x, y

def train_and_predict(X_in, y_in, unlabeled_ids, clf=None):
    """Train the model.

    Parameters
    ----------
    X_in : dict of DataFrames
        keys correspond to unlabeled_ids
    y_in : DataFrame
        [id, label] with multi-class labels [0, 1, 2, ...]
    unlabeled_ids : Series
        ids of unlabeled companies
    n_neighbors : optional (default=5)
        number of neighbors to use in the kNN-classifier
    """
    # Model the outcome
    # C = {}
    pred = pd.DataFrame()
    score = pd.Series()
    fm = pd.Series()
    f1 = pd.DataFrame()
    count = 0

    if clf is None:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    # One-hot encode labels for kNN classifier
    yb = known_one_hot(y_in)

    for _, ul in unlabeled_ids.iteritems():
        # Split out unknown and get similarity of single vector
        X_aug = X_in[ul]
        s = X_aug.loc[X_aug.id == ul, feat_cols].copy() # (m, 1) features of unknown company
        X = X_aug.loc[X_aug.id != ul, feat_cols].copy() # (Nl, m) features of labeled companies

        # Simple model:
        # C[ul] = similarity(s, X) # (m, 1) similarity vector
        # # To predict: use single nearest neighbor with cosine similarity
        # idx = C[ul].values.argsort(axis=0).squeeze()[::-1] # array shape (m,) descending
        # sim_idx = C[ul].iloc[idx[0:6]].index
        # pred[ul] = y_in.loc[sim_idx[0]]

        X, y = upsample_minority(X, y_in, yb, min_lab=3)

        # Don't use "train", "test" here to avoid over-writing previous time break
        X_tr, X_t, y_tr, y_t = train_test_split(X, y, train_size=0.6, 
                                                stratify=y,
                                                random_state=56)

        # TODO try a couple classifiers? SVM w/ RBF?
        # More advanced model: let k-NN do the work
        clf.fit(X_tr, y_tr)

        # Mean accuracy over all classes
        score_i = clf.score(X_t, y_t)
        y_hat = clf.predict(X_t)
        fm_i = precision_recall_fscore_support(y_t, y_hat, average='macro')
        f1_i = precision_recall_fscore_support(y_t, y_hat, labels=[0, 1, 2, 3], average=None)

        # Store in output dataframes
        score = score.append(pd.Series({ul:score_i}))
        fm = fm.append(pd.Series({ul:fm_i}))
        f1 = f1.append(pd.DataFrame.from_dict({ul:f1_i}, orient='index'))

        # predict for the single unknown company!
        pred_i = clf.predict(s).squeeze() # just get single vector
        pred = pred.append(pd.DataFrame.from_dict({ul:pred_i}, orient='index'))

        count += 1
        if count > MAX_COUNT:
            break

    return pred, score, fm, f1

def upsample_minority(X_in, y_in, yb, min_lab=3, n_classes=4):
    """Create upsampled versions to balance classes."""
    # Upsample minority classes
    X_maj = X_in.loc[y_in.label == 3] # majority (y_in.label.value_counts())
    y_maj = yb.loc[y_in.label == 3]
    X_min_u = []
    y_min_u = []
    for i in range(n_classes-1):
        # Upsample minority class
        X_min_u.append(resample(X_in.loc[y_in.label == i], 
                            replace=True, 
                            n_samples=X_maj.shape[0],
                            random_state=56))
        y_min_u.append(resample(yb.loc[y_in.label == i], 
                            replace=True, 
                            n_samples=X_maj.shape[0],
                            random_state=56))

    # Combine majority class with upsampled minority class
    X = pd.concat([X_maj] + X_min_u)
    y = pd.concat([y_maj] + y_min_u)
    return X, y

#==============================================================================
#==============================================================================
