#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: cv_plots.py
#  Created: 06/28/2018, 13:25
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import itertools
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from time import time
import seaborn as sns
from scipy import interp

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from timeline_features import classes, upsample_minority

save_flag = 0
fig_dir = '../figures/'
fig_ext = '.png'

plt.ion()
plt.close('all')
sns.set_style('whitegrid')
sns.set_context('poster', font_scale=1.3)
# sns.set_context()

feat_cols = ['experience', 'latitude', 'longitude', 'offices', 'products',
             'mean_fund_time', 'funding_rounds', 'mean_funding_amt',
             'cumulative_famt', 'mean_milestone_time', 'milestones',
             'mean_investment_time', 'investments', 'mean_acquisition_time',
             'acquisitions']

#------------------------------------------------------------------------------ 
#       Load the data
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
# X_max, y_max, X, y = pickle.load(open('../data/Xy_age_5yrs.pkl', 'rb'))

# Binarize the labels
lb = LabelBinarizer()
yb = lb.fit_transform(y)
yb = pd.DataFrame(data=yb, index=y.index)

clf = RandomForestClassifier(criterion='entropy', max_depth=None, 
                             max_features=4, min_samples_split=3, n_jobs=10)

X_train, X_test, y_train, y_test = train_test_split(X, yb, train_size=0.6, 
                                                    stratify=yb,
                                                    random_state=56)
clf.fit(X_train, y_train)

def class_rpt(X, y):
    p = clf.predict(X)
    p = pd.DataFrame(data=p, index=y.index)
    print(classification_report(y, p))
    return p

p_train = class_rpt(X_train, y_train)
p_test = class_rpt(X_test, y_test)

if hasattr(clf, 'feature_importances_'):
    fp = dict(zip(feat_cols, clf.feature_importances_))
    fp = [(k, fp[k]) for k in sorted(fp, key=fp.get, reverse=True)]

#------------------------------------------------------------------------------ 
#        Pair plot of features
#------------------------------------------------------------------------------
# df_pairs = X_max.merge(y_max, on='id', how='inner')
# for i in range(0, len(feat_cols), 5):
#     g = sns.pairplot(data=df_pairs,
#                     vars=feat_cols[i:i+5],
#                     hue='label')

#------------------------------------------------------------------------------ 
#        Confusion matrix
#------------------------------------------------------------------------------
cm = confusion_matrix(lb.inverse_transform(y_test.values), 
                      lb.inverse_transform(p_test.values))

normalize=True
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(2, figsize=(9, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix: Company Age ~ 5.5 years')
plt.grid('off')
cbar = plt.colorbar(shrink=0.8)
# v = np.linspace(0, 1, 6)
# cbar = plt.colorbar(shrink=0.8, ticks=v)
# cbar.ax.set_xticklabels([str(x) for x in v])
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
if save_flag:
    plt.savefig(fig_dir + 'confusion_5y' + fig_ext)

#------------------------------------------------------------------------------ 
#        ROC Curves
#------------------------------------------------------------------------------
# Probabilities for each class
probas = clf.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
n_classes = yb.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test.loc[:, i], probas[i][:, 1])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
# probas_l = np.array([x[:, 1] for x in probas]).T
# fpr['micro'], tpr['micro'], _ = roc_curve(y_test.values.ravel(), probas_l.ravel())
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
plt.figure(56, figsize=(9, 8))
plt.clf()

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], 
             label="{} (AUC = {:0.2f})".format(classes[i], roc_auc[i]))

plt.plot(fpr['macro'], tpr['macro'],
         label="Macro-average (AUC = {:0.2f})".format(roc_auc['macro']),
         color='navy', linestyle='-.')

# plt.plot(fpr['micro'], tpr['micro'],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc['micro']),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=1) # random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC: Company Age ~ 5.5 years')
plt.legend()
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'ROC_5y' + fig_ext)

# #------------------------------------------------------------------------------ 
# # TODO       Plot decision boundary
# #------------------------------------------------------------------------------
# # Visualize result after dimensionality reduction using truncated SVD
# svd = TruncatedSVD(n_components=2)
# X_reduced = svd.fit_transform(X)
#
# # scatter plot of original and reduced data
# fig = plt.figure(1, figsize=(9, 8))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y.squeeze(), s=50, edgecolor='k')
# # plt.scatter(X['mean_fund_time'], X['mean_funding_amt'], c=y.squeeze(), s=50, edgecolor='k')
# # plt.title('Truncated SVD reduction: Company Age ~ 5.5 years')
#
# clf.fit(X_reduced, yb)
#
# h = 0.2  # step size in the mesh
# x_min, x_max = X.values[:, 0].min() - .5, X.values[:, 0].max() + .5
# y_min, y_max = X.values[:, 1].min() - .5, X.values[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]*[y_min, y_max].
# if hasattr(clf, 'decision_function'):
#     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# else:
#     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
#
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xticks(())
# ax.set_yticks(())
#
# if save_flag:
#     plt.savefig(fig_dir + 'SVD_5y' + fig_ext)

plt.show()
#==============================================================================
#==============================================================================
