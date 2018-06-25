#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: eda_timeseries.py
#  Created: 06/24/2018, 18:42
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
import matplotlib.dates as mdates
from scipy import interp
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample

plt.ion()
plt.close('all')

#------------------------------------------------------------------------------ 
#        IMPORT THE DATA!!
#------------------------------------------------------------------------------
filename = '../data/cb_input_datasets.pkl'
tf, df = pickle.load(open(filename, 'rb'))

# Plot events per month
# tf.groupby([tf.dates.dt.year, tf.dates.dt.month]).count().plot(kind='bar')
fig, ax = plt.subplots(figsize=(11,9))
tf.dates.groupby(tf.dates.dt.year).count().plot(ax=ax, kind='bar', color='C0')

# ax.xaxis.set_major_locator(mdates.YearLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax.set_xlim(np.datetime64('1990', 'Y'), np.datetime64('2014', 'Y'))

plt.grid('off')
plt.tight_layout()
plt.show()

#==============================================================================
#==============================================================================
