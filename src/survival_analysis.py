#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: survival_analysis.py
#  Created: 06/20/2018, 18:59
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, CoxTimeVaryingFitter, AalenAdditiveFitter
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import k_fold_cross_validation
from lifelines.utils import to_long_format, add_covariate_to_timeline

# Load the data
lf = pd.read_pickle('../data/survival_input_clean.pkl')

# Drop rows where start == stop == 0
lf = lf.loc[~((lf.start == lf.stop) & (lf.start == 0))]
lf = lf.loc[~((lf.start < 0) | (lf.stop < 0))]

# Cut off study at t = 15 years
# lf.loc[lf.stop > 15*365, 'stop'] = 15*365
lf.start.fillna(value=0, inplace=True)
lf.fillna(value=lf.median(), inplace=True)

# Randomly take small sample for testing
# test = lf.loc[lf.id.isin(np.random.choice(lf.id, size=1000))]

# lr = lf[(lf.name == 'Twitter') | (lf.name == 'Facebook')]

# Using Cox Proportional Hazards model
# cph = CoxPHFitter()
# drop_cols = ['category_code', 'name', 'id', 'founded_at']
# lf_n = test.drop(columns=drop_cols)
# cph.fit(lf_n, duration_col='stop', event_col='success')
# cph.print_summary()
# cph.plot()

# ctv = CoxTimeVaryingFitter()
# ctv.fit(lf, id_col='id', event_col='success', start_col='start', stop_col='stop',
#         show_progress=True, step_size=0.1)
# ctv.print_summary()
# ctv.plot()

# Fit the survival curve
kmf = KaplanMeierFitter()
kmf.fit(durations=lf.stop, event_observed=lf.success)  # or, more succiently, kmf.fit(T, E)
kmf.plot()

plt.show()
#==============================================================================
#==============================================================================
