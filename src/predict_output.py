#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: predict_output.py
#  Created: 06/13/2018, 15:11
#   Author: Bernie Roesler
#
"""
  Description: Predict outcomes using final model version.
  We will take values learned from `train_model.py` and implement them in this
  script to interact with the MVP webapp.
"""
#==============================================================================

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# IMPORT THE DATA!!
df = pd.read_csv('../data/cb_input.csv', index_col='id')

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'experience']
labels = ['success', 'failure']

X = df[feat_cols]
y = df[labels]

# Impute NaN values to mean of column
X = X.fillna(X.median())

# Scale the data
Xn = StandardScaler().fit_transform(X)
Xn = pd.DataFrame(data=Xn, columns=X.columns, index=X.index)

# Test/Train Split
X_train, X_test, y_train, y_test = train_test_split(Xn, y, 
                                                    train_size=0.6, 
                                                    random_state=56)

#------------------------------------------------------------------------------ 
#        Train the Model
#------------------------------------------------------------------------------
rfc = RandomForestClassifier(n_estimators=10)
rfc = rfc.fit(X_train, y_train)

# Predict for entire database
pred = rfc.predict(Xn)
pred = pd.DataFrame(data=pred, columns=y.columns, index=y.index)

# TODO incorporate this code into the callback:
in_str = 'ebay' # instagarage, ebay
in_permalink = '/company/' + in_str
out_id = df[df.permalink == in_permalink].index[0]
out_pred = pred.loc[out_id]
out_info = df.loc[out_id]


#==============================================================================
#==============================================================================
