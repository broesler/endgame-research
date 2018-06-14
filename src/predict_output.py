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
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#------------------------------------------------------------------------------ 
#        Define helper functions
#------------------------------------------------------------------------------
def Nstats(X, y):
    """Return tuple of fractions for given dataset."""
    N = X.shape[0] # total number of examples
    Fs = y[(y.success == 1) & (y.failure == 0)].shape[0] / N # fraction success
    Ff = y[(y.success == 0) & (y.failure == 1)].shape[0] / N # fraction failure
    Fu = y[(y.success == 0) & (y.failure == 0)].shape[0] / N # fraction unknown
    return np.array([N, Fs, Ff, Fu])

# Equivalent to "shortg" Matlab format
np.set_printoptions(precision=4, suppress=True)

plt.ion()

# IMPORT THE DATA!!
df = pd.read_csv('../data/my_cb_input.csv', index_col='id')

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'experience']
labels = ['success', 'failure']

X = df[feat_cols]
y = df[labels]

Nt = Nstats(X, y)

# Impute NaN values to mean of column
X = X.fillna(X.median())

# Scale the data
Xn = StandardScaler().fit_transform(X)
Xn = pd.DataFrame(data=Xn, columns=X.columns, index=X.index)

# Test/Train Split
X_train, X_test, y_train, y_test = train_test_split(Xn, y, train_size=0.7, random_state=56)

N_train = Nstats(X_train, y_train)
N_test = Nstats(X_test, y_test)

#------------------------------------------------------------------------------ 
#        Train the Model
#------------------------------------------------------------------------------
rfc = RandomForestClassifier(n_estimators=10)
rfc = rfc.fit(X_train, y_train)

# Predict for entire database
pred = rfc.predict(Xn)
pred = pd.DataFrame(data=pred, columns=y.columns, index=y.index)

# | Instagarage | Instagarage.com is an online company that will provide
# a universal gift card where all gift cards are built into one single card
# that will allow consumers to shop at the retailer of their choice. The
# Instagarage card will allow consumers to choose multiple retailers, select
# the dollar amount for each retailer, customize their card, and place them all
# on a universal gift card to be used at multiple stores on or offline. The
# universal gift card provides for a fast and easy way to purchase merchandise
# online, via a cellphone display screen, or at the retailer's checkout
# counter.
# Closed: February 2, 2011

in_str = 'instagarage' # instagarage, ebay
in_permalink = '/company/' + in_str

out_id = df[df.permalink == in_permalink].index[0]
out_pred = pred.loc[out_id]
out_info = df.loc[out_id]

#==============================================================================
#==============================================================================
