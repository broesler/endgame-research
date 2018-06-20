#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: lifelines_regression_ex.py
#  Created: 06/19/2018, 20:44
#   Author: Bernie Roesler
#
"""
  Description: Regression example with covariates
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines.datasets import load_regression_dataset
from lifelines import CoxPHFitter, AalenAdditiveFitter
from lifelines.utils import k_fold_cross_validation

plt.close('all')

regression_dataset = load_regression_dataset()
print(regression_dataset.head())

# Using Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(regression_dataset, duration_col='T', event_col='E')
cph.print_summary()

"""
n=200, number of events=189

       coef  exp(coef)  se(coef)      z      p  lower 0.95  upper 0.95
var1 0.2213     1.2477    0.0743 2.9796 0.0029      0.0757      0.3669  **
var2 0.0509     1.0522    0.0829 0.6139 0.5393     -0.1116      0.2134
var3 0.2186     1.2443    0.0758 2.8836 0.0039      0.0700      0.3672  **
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Concordance = 0.580
"""

cph.plot()

# Using Aalen's Additive model
aaf = AalenAdditiveFitter(fit_intercept=False)
aaf.fit(regression_dataset, duration_col='T', event_col='E')
aaf.plot()

X = regression_dataset.drop(['E', 'T'], axis=1)
aaf.predict_survival_function(X.iloc[10:12]).plot()  # get the unique survival functions of two subjects

scores = k_fold_cross_validation(cph, regression_dataset, duration_col='T', event_col='E', k=10)
print(scores)
print(np.mean(scores))
print(np.std(scores))

plt.show()
#==============================================================================
#==============================================================================
