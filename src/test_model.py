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
from matplotlib import rc as mplrc
from mpl_toolkits.mplot3d import axes3d
from scipy import interp
import seaborn as sns

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample

from sklearn.metrics.pairwise import cosine_similarity

# Equivalent to "shortg" Matlab format
np.set_printoptions(precision=4, suppress=True)

plt.ion()

#------------------------------------------------------------------------------ 
#        IMPORT THE DATA!!
#------------------------------------------------------------------------------
df = pd.read_pickle('../data/cb_input_multi.pkl')

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'avg_time_to_funding',
             'experience', 'a', 'angel', 'b', 'c', 'convertible', 'crowd',
             'crowd_equity', 'd', 'debt_round', 'e', 'f', 'g', 'grant',
             'partial', 'post_ipo_debt', 'post_ipo_equity', 'private_equity',
             'secondary_market', 'seed', 'unattributed']

classes = ['Failed', 'Timely Exit', 'Late Exit', 'Steady Operation']

#------------------------------------------------------------------------------ 
#        Upsample minority classes
#------------------------------------------------------------------------------
# >>> df.label.value_counts()
# 3.0    16904
# 1.0     2103
# 0.0     1178
# 2.0      683
# Name: label, dtype: int64

df_maj = df[df.label == 3] # majority class
df_min_u = []
for i in range(3):
    # Upsample minority class
    df_min_u.append(resample(df[df.label == i], 
                           replace=True, 
                           n_samples=df_maj.shape[0],
                           random_state=56))

# Combine majority class with upsampled minority class
df_up = pd.concat([df_maj] + df_min_u)

# Display new class counts
# df_up.label.value_counts()
# 1.0    16904
# 2.0    16904
# 0.0    16904
# 3.0    16904
# Name: label, dtype: int64

#------------------------------------------------------------------------------ 
#        Extract features and labels
#------------------------------------------------------------------------------
X = df_up[feat_cols]

# Binarize the labels
y = df_up.label
yb = label_binarize(y, classes=[0, 1, 2, 3])
y = pd.DataFrame(data=yb, index=y.index)

#------------------------------------------------------------------------------ 
#        Normalize the data
#------------------------------------------------------------------------------
# Impute NaN values to mean of column
X = X.fillna(X.median())

# Scale the data
Xn = StandardScaler().fit_transform(X)
Xn = pd.DataFrame(data=Xn, columns=X.columns, index=X.index)

# Train/Test Split -- "_test" == HOLDOUT
X_train, X_test, y_train, y_test = train_test_split(Xn, y, 
                                                   train_size=0.9, 
                                                    stratify=y,
                                                    random_state=56)

# Split into training/cross-validation 
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, 
                                                train_size=0.6, 
                                                stratify=y_train,
                                                random_state=56)

#------------------------------------------------------------------------------ 
#        Train the Model
#------------------------------------------------------------------------------
clf = RandomForestClassifier(min_samples_split=50)
clf = clf.fit(X_train, y_train)

pred_train = clf.predict(X_train)
pred_train = pd.DataFrame(data=pred_train,
                          index=y_train.index)
print(classification_report(y_train, pred_train))

pred_cv = clf.predict(X_cv)
pred_cv = pd.DataFrame(data=pred_cv,
                          index=y_cv.index)
print(classification_report(y_cv, pred_cv))

pred_test = clf.predict(X_test)
pred_test = pd.DataFrame(data=pred_test,
                         index=y_test.index)
# print(classification_report(y_test, pred_test))

#------------------------------------------------------------------------------ 
#        Feature Importance
#------------------------------------------------------------------------------
fp = dict(zip(feat_cols, clf.feature_importances_))
fp = [(k, fp[k]) for k in sorted(fp, key=fp.get, reverse=True)]

ranked_features = [x[0] for x in fp]

# Get predictions for all companies (not using upsampled data)
Xa = df[feat_cols]
Xa = Xa.fillna(Xa.median())
Xan = StandardScaler().fit_transform(Xa)
Xan = pd.DataFrame(data=Xan, columns=Xa.columns, index=Xa.index)

# Predict ALL outcomes (not using upsampled data)
pred = clf.predict(Xan)
pred = pd.DataFrame(data=pred, index=Xan.index)
pred_probas = clf.predict_proba(Xan)

# PICKLE ME
pickle.dump([pred, pred_probas], open('../data/predictions.pkl', 'wb'))
# pred, pred_probas = pickle.load(open('../data/predictions.pkl', 'rb'))

# Get most similar companies
# Xan = Xan[df.category_code == 'web']
# C = cosine_similarity(Xan[ranked_features[0:3]])
# C = pd.DataFrame(data=C, index=Xan.index, columns=Xan.index).abs()

#------------------------------------------------------------------------------ 
#        ROC Curves
#------------------------------------------------------------------------------
probas_ = clf.predict_proba(X_test)

# To test multiple classifiers:
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# if hasattr(clf, "decision_function"):
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# else:
#     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Compute ROC curve and ROC area for each class
n_classes = y.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test.loc[:, i], probas_[i][:, 1])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
# fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

# Plot all ROC curves
plt.figure(1)
plt.clf()

# plt.plot(fpr['micro'], tpr['micro'],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc['micro']),
#          color='deeppink', linestyle=':', linewidth=4)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], 
             label="{} (AUC = {:0.2f})".format(classes[i], roc_auc[i]))

plt.plot(fpr['macro'], tpr['macro'],
         label="Macro-average (AUC = {:0.2f})".format(roc_auc['macro']),
         color='navy', linestyle='-.')

plt.plot([0, 1], [0, 1], 'k--', lw=1) # random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic')
plt.legend()
plt.tight_layout()
plt.show()

#==============================================================================
#==============================================================================
