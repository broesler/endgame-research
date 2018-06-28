#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: output_plots.py
#  Created: 06/27/2018, 17:00
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from timeline_features import get_sorted_values

save_flag = 0
fig_dir = '../figures/'
fig_ext = '.png'

pred_test, ages_test, score_test, f1_test, fm_test = \
    pickle.load(open('../data/timeline_output_test_full.pkl', 'rb'))
    # pickle.load(open('../data/timeline_output_test.pkl', 'rb'))

n_neighbors = 5

#------------------------------------------------------------------------------ 
#        Plot accuracy vs company age
#------------------------------------------------------------------------------
ages_t, score_t = get_sorted_values(ages_test, score_test)
_, fm_t = get_sorted_values(ages_test, fm_test)
max_age = ages_t.max() / 365

# On originally labeled companies
plt.figure(1)
plt.clf()
plt.plot([0, max_age], [0.25, 0.25], 'k--', label='Baseline')
# "score" is same as recall
plt.plot(ages_t / 365, score_t,
         label='Mean Accuracy: $k$-NN, $k$ = {}'.format(n_neighbors))
plt.grid('on')
plt.xlabel('Company Age [years]')
plt.ylabel('Classifier Score')
plt.ylim([0, 1])
plt.legend()
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'score_vs_age' + fig_ext)

# Plot precision and recall
plt.figure(2)
plt.clf()
plt.plot([0, max_age], [0.25, 0.25], 'k--', label='Baseline')
plt.plot(ages_t / 365, fm_t[:, 0], label='Precision')
plt.plot(ages_t / 365, fm_t[:, 1], label='Recall')
plt.plot(ages_t / 365, fm_t[:, 2], label='$F_1$ Score')
plt.grid('on')
plt.title('$k$-NN, $k$ = 5')
plt.xlabel('Company Age [years]')
plt.ylabel('Classifier Score')
plt.ylim([0, 1])
plt.legend()
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'score_vs_age' + fig_ext)

plt.show()


#==============================================================================
#==============================================================================
