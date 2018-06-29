#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: timeseries_error_plot.py
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

def get_sorted_values(dx, dy):
    """Sort dicts by dx.items() and return arrays of dx and dy values."""
    sort_keys = [x[0] for x in sorted(dx.items(), key=lambda kv: kv[1])]
    x = np.array([dx[k] for k in sort_keys])
    y = np.array([dy[k] for k in sort_keys])
    return x, y

save_flag = 1
fig_dir = '../figures/'
fig_ext = '.png'

plt.ion()

N = 50
filename = '../data/timeline_output_rf{}.pkl'.format(N)
pred, ages, unlabeled_ids, fm = pickle.load(open(filename, 'rb'))

#------------------------------------------------------------------------------ 
#        Plot accuracy vs company age
#------------------------------------------------------------------------------
ages_t, fm_t = get_sorted_values(ages[unlabeled_ids[:N]], fm)
max_age = ages_t.max() / 365

# On originally labeled companies
plt.figure(1)
plt.clf()
plt.plot([0, max_age], [1/3, 1/3], 'k--', label='Baseline')
plt.plot(ages_t / 365, fm_t[:, 2], label='Random Forest')
plt.grid('on')
plt.xlabel('Company Age [years]')
plt.ylabel('$F_1$ Score')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'f1_vs_age_rf' + fig_ext)

# Plot precision and recall
plt.figure(2)
plt.clf()
plt.plot([0, max_age], [1/3, 1/3], 'k--', label='Baseline')
plt.plot(ages_t / 365, fm_t[:, 2], label='$F_1$ Score')
plt.plot(ages_t / 365, fm_t[:, 0], label='Precision')
plt.plot(ages_t / 365, fm_t[:, 1], label='Recall')
plt.grid('on')
# plt.title('$k$-NN, $k$ = 5')
plt.xlabel('Company Age [years]')
plt.ylabel('Classifier Score')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.tight_layout()
if save_flag:
    plt.savefig(fig_dir + 'prf_vs_age_rf' + fig_ext)

plt.show()


#==============================================================================
#==============================================================================
