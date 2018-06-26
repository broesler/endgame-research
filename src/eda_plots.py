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

save_flag = 0
fig_dir = '../figures/'
fig_ext = '.png'

plt.close('all')
sns.set_style('whitegrid')
sns.set_context('poster')

np.set_printoptions(precision=4, suppress=True) # "shortg" Matlab format
plt.ion()

# IMPORT THE DATA!!
df = pd.read_pickle('../data/cb_input_multi.pkl')

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'avg_time_to_funding',
             'experience']
# funding_rounds = ['a', 'angel', 'b', 'c', 'convertible', 'crowd',
#              'crowd_equity', 'd', 'debt_round', 'e', 'f', 'g', 'grant',
#              'partial', 'post_ipo_debt', 'post_ipo_equity', 'private_equity',
#              'secondary_market', 'seed', 'unattributed']

classes = ['Failed', 'Timely Exit', 'Late Exit', 'Steady Operation']
colors = ['C3', 'C2', 'C1', 'C0']

df['age_at_exit_years'] = df.age_at_exit / 365

y = df.label

#------------------------------------------------------------------------------ 
#        Data Statistics
#------------------------------------------------------------------------------
plt.figure(1, figsize=(11, 9))
plt.clf()
to_plot = ['web', 'transportation', 'health'] # select categories
g = sns.boxplot(y='category_code', 
                x='age_at_exit_years', 
                data=df[(y == 1) & (df.category_code.isin(to_plot))])
plt.title('Age at Exit (Acquisition or IPO)')
plt.xticks(rotation=70)
g.set_xlabel('Age (years)')
g.set_ylabel('Industry')
# g.set_xlim([0, 50])
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'age_at_exit_success' + fig_ext)

# plt.figure(2, figsize=(11, 9))
# plt.clf()
# g = sns.boxplot(y='category_code', 
#                 x='age_at_exit_years', 
#                 data=df[y == 0])
# plt.title('Age at Close')
# g.set_xlabel('Age (years)')
# g.set_ylabel('Category')
# g.set_xlim([0, 50])
# plt.xticks(rotation=70)
# plt.tight_layout()
# if save_flag:
#     plt.savefig(fig_dir + 'age_at_exit_failure' + fig_ext)

# plt.figure(5, figsize=(11, 9))
# plt.clf()
# g = sns.boxplot(y='category_code', 
#                 x='age_at_exit_years', 
#                 data=df[y == 0])
# plt.title('Age at Close')
# g.set_xlabel('Age (years)')
# g.set_ylabel('Category')
# g.set_xlim([0, 50])
# plt.xticks(rotation=70)
# plt.tight_layout()
# if save_flag:
#     plt.savefig(fig_dir + 'age_at_exit_failure' + fig_ext)

# plt.figure(3, figsize=(11, 9))
# plt.clf()
# g = sns.boxplot(y='category_code', 
#                 x='age_at_exit_years', 
#                 data=df[y == 2])
# plt.title('Age of Dinosaurs')
# g.set_xlabel('Age (years)')
# g.set_ylabel('Category')
# g.set_xlim([0, 50])
# plt.xticks(rotation=70)
# plt.tight_layout()
# if save_flag:
#     plt.savefig(fig_dir + 'age_at_exit_lateexit' + fig_ext)
#
# plt.figure(4, figsize=(11, 9))
# plt.clf()
# g = sns.boxplot(y='category_code', 
#                 x='age_at_exit_years', 
#                 data=df[y == 3])
# plt.title('Age of Operating Companies')
# g.set_xlabel('Age (years)')
# g.set_ylabel('Category')
# g.set_xlim([0, 50])
# plt.xticks(rotation=70)
# plt.tight_layout()
# if save_flag:
#     plt.savefig(fig_dir + 'age_at_exit_operating' + fig_ext)

# #------------------------------------------------------------------------------ 
# #        Plot location
# #------------------------------------------------------------------------------
# plt.figure()
# plt.clf()
# ax = plt.gca()
# for i in range(len(classes)):
#     ax.scatter(df.loc[df.label == i, 'longitude'],
#                df.loc[df.label == i, 'latitude'], 
#                s=10, color=colors[i], label=classes[i])
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Company Location')
# ax.legend()
# plt.tight_layout()
# if save_flag:
#     plt.savefig(fig_dir + 'location' + fig_ext)

# #------------------------------------------------------------------------------ 
# #        Plot Correlation Between Features
# #------------------------------------------------------------------------------
# corr = df[feat_cols].corr()
#
# # Mask off upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# # Plot it
# fig, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# sns.heatmap(corr, mask=mask, cmap=cmap, 
#             vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=0.5)
# plt.tight_layout()
# if save_flag:
#     plt.savefig(fig_dir + 'corr_mat' + fig_ext)

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
