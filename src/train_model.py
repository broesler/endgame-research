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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
from mpl_toolkits.mplot3d import axes3d
from scipy import interp
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve

from cb_funcs import Nstats

# Equivalent to "shortg" Matlab format
np.set_printoptions(precision=4, suppress=True)

plt.ion()

# IMPORT THE DATA!!
df = pd.read_pickle('../data/cb_input.pkl')

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'experience']
labels = ['success', 'failure']

X = df[feat_cols]
# y = df[labels]
y = df['bin_label']

#------------------------------------------------------------------------------ 
#        Data Statistics
#------------------------------------------------------------------------------
# Nt = Nstats(X, y)

# Impute NaN values to mean of column
X = X.fillna(X.median())

# Scale the data
Xn = StandardScaler().fit_transform(X)
Xn = pd.DataFrame(data=Xn, columns=X.columns, index=X.index)

# Test/Train Split
X_train, X_test, y_train, y_test = train_test_split(Xn, y, 
                                                    train_size=0.6, 
                                                    stratify=y,
                                                    random_state=56)

# N_train = Nstats(X_train, y_train)
# N_test = Nstats(X_test, y_test)

# # Perform PCA decomposition to see n most important stats
# U, sigma, V = np.linalg.svd(Xn.T) # sigma is shape (m,) array, NOT matrix
#
# m = sigma.shape[0]  # number of singular values
# k = 6               # number of dimensions in subspace (i.e. PCs to keep)
#
# explained_variance_ratio = sigma / sum(sigma)
#
# thresh = 0.8
# idx_sig = np.argmax(np.cumsum(explained_variance_ratio) >= thresh)
# sigma_allowed = sigma[:idx_sig]

#------------------------------------------------------------------------------ 
#        Train the Model
#------------------------------------------------------------------------------
# rfc = RandomForestClassifier(n_estimators=10, class_weight='balanced')
rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
# rfc = rfc.fit(X_train, y_train)

pred_train = rfc.predict(X_train)
pred_train = pd.DataFrame(data=pred_train,
                          index=y_train.index)
print(classification_report(y_train, pred_train))

pred_test = rfc.predict(X_test)
pred_test = pd.DataFrame(data=pred_test,
                         index=y_test.index)
print(classification_report(y_test, pred_test))

pred = rfc.predict(Xn)
pred = pd.DataFrame(data=pred, index=y.index)

# Confusion matrix -- need to convert back to single value encoding
# cm = pd.DataFrame(confusion_matrix(y_test, pred_test).T,
#                   index=['Yes', 'No'],
#                   columns=['Yes', 'No'])
# cm.index.name = 'Predicted'
# cm.columns.name = 'True'
# cm

#------------------------------------------------------------------------------ 
#        Feature Importance
#------------------------------------------------------------------------------
fp = dict(zip(feat_cols, rfc.feature_importances_))
fp = [(k, fp[k]) for k in sorted(fp, key=fp.get, reverse=True)]

#------------------------------------------------------------------------------ 
#        ROC Curves
#------------------------------------------------------------------------------
tprs = []
mean_fpr = np.linspace(0, 1, 100)

probas_ = rfc.predict_proba(X_test)
# Compute ROC curve and area the curve
fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0

plt.figure(1)
plt.clf()
plt.plot(fpr, tpr, label='Random Forest')
plt.plot([0, 1], [0, 1], 
         linestyle='--', lw=2, color='k', alpha=0.8, 
         label='Pure Chance')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

#==============================================================================
#==============================================================================
