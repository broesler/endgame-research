#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: eda_plots.py
#  Created: 06/13/2018, 15:11
#   Author: Bernie Roesler
#
"""
  Description: Exploratory Data Analysis plots with given features
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

plt.ion()

# IMPORT THE DATA!!
# from build_features import df

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'experience']
labels = ['label']

#------------------------------------------------------------------------------ 
#        Data Statistics
#------------------------------------------------------------------------------
N = df.shape[0]
Nf = df[df.label == 0].shape[0] / N      # fraction failure
Ne = df[df.label == 1].shape[0] / N      # fraction success
Nu = df[df.label == 2].shape[0] / N      # fraction unknown

#------------------------------------------------------------------------------ 
#        Plot acquisition time vs category label
#------------------------------------------------------------------------------
# rf = df[['category_code', 'label']].groupby('category_code').sum()
plt.figure(1)
g = sns.boxplot(x='category_code', y='label', data=df)
plt.xticks(rotation=70)
plt.tight_layout()
#------------------------------------------------------------------------------ 
#        Pair plot of features
#------------------------------------------------------------------------------
# mplrc('text', usetex=False)
plt.figure(2)
g = sns.pairplot(df[feat_cols[0:5]])

# minmax = pd.DataFrame([df[feat_cols].max(axis=0), df[feat_cols].min(axis=0)])

#==============================================================================
#==============================================================================
