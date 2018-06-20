#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: lifelines_ex.py
#  Created: 06/19/2018, 20:31
#   Author: Bernie Roesler
#
"""
  Description: lifelines survival analysis example.
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines.datasets import load_waltons
from lifelines import KaplanMeierFitter, NelsonAalenFitter

plt.close('all')

df = load_waltons() # returns a Pandas DataFrame

print(df.head())
"""
    T  E    group
0   6  1  miR-137
1  13  1  miR-137
2  13  1  miR-137
3  13  1  miR-137
4  19  1  miR-137
"""

T = df['T']
E = df['E']

# Fit the survival curve
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)  # or, more succiently, kmf.fit(T, E)
kmf.plot()

# Plot cumulative hazard function
naf = NelsonAalenFitter()
naf.fit(T, E)
naf.plot()

#------------------------------------------------------------------------------ 
#        Multiple groups
#------------------------------------------------------------------------------
groups = df['group']
ix = (groups == 'miR-137')

kmf.fit(T[~ix], E[~ix], label='control')
ax = kmf.plot()

kmf.fit(T[ix], E[ix], label='miR-137')
kmf.plot(ax=ax)

plt.show()
#==============================================================================
#==============================================================================
