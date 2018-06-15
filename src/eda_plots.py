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
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from cb_funcs import Nstats

save_flag = 1
fig_dir = './figures/'
fig_ext = '.png'

plt.close('all')
sns.set_style('whitegrid')
sns.set_context('talk')

np.set_printoptions(precision=4, suppress=True) # "shortg" Matlab format
plt.ion()

# IMPORT THE DATA!!
df = pd.read_csv('../data/test_cb_input.csv', index_col='id')

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'experience']
labels = ['success', 'failure']

df['age_at_exit_years'] = df.age_at_exit / 365

X = df[feat_cols]
y = df[labels]

#------------------------------------------------------------------------------ 
#        Data Statistics
#------------------------------------------------------------------------------
Nt = Nstats(X, y)

plt.figure(1, figsize=(11, 9))
plt.clf()
g = sns.boxplot(x='category_code', 
                y='age_at_exit_years', 
                data=df[y.success == 1])
plt.title('Age at Exit (Acquisition or IPO)')
plt.xticks(rotation=70)
g.set_ylabel('Age (years)')
g.set_xlabel('Category')
g.set_ylim([0, 50])
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'age_at_exit_success' + fig_ext)

plt.figure(2, figsize=(11, 9))
plt.clf()
g = sns.boxplot(x='category_code', 
                y='age_at_exit_years', 
                data=df[y.failure == 1])
plt.title('Age at Close')
g.set_ylabel('Age (years)')
g.set_xlabel('Category')
g.set_ylim([0, 50])
plt.xticks(rotation=70)
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'age_at_exit_failure' + fig_ext)

plt.figure(3, figsize=(11, 9))
plt.clf()
g = sns.boxplot(x='category_code', 
                y='age_at_exit_years', 
                data=df[(y.success == 0) & (y.failure == 0)])
plt.title('Age of Operating Companies')
g.set_ylabel('Age (years)')
g.set_xlabel('Category')
g.set_ylim([0, 50])
plt.xticks(rotation=70)
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'age_at_exit_operating' + fig_ext)

#------------------------------------------------------------------------------ 
#        Plot location
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------ 
#        Plot Correlation Between Features
#------------------------------------------------------------------------------
corr = X.corr()

# Mask off upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot it
fig, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, 
            vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=0.5)
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'corr_mat' + fig_ext)

#------------------------------------------------------------------------------ 
#        Pair plot of features
#------------------------------------------------------------------------------
# Impute NaN values to mean of column
# dff = df.fillna(df.median())

# for i in range(0, len(feat_cols), 5):
#     g = sns.pairplot(data=dff,
#                     vars=feat_cols[i:i+5],
#                     hue='label')

#==============================================================================
#==============================================================================
