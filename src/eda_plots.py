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
from build_features import df, N

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'usd_per_round', 'experience']
labels = ['label']

#------------------------------------------------------------------------------ 
#        Data Statistics
#------------------------------------------------------------------------------
# N = df.shape[0]
Nf = df[df.label == 0].shape[0] / N      # fraction failure
Ne = df[df.label == 1].shape[0] / N      # fraction success
Nu = df[df.label == 2].shape[0] / N      # fraction unknown

#------------------------------------------------------------------------------ 
#        Pair plot of features
#------------------------------------------------------------------------------
mplrc('text', usetex=False)
sns.pairplot(df[feat_cols[0:5]])

#==============================================================================
#==============================================================================
