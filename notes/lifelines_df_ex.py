#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: lifelines_df_ex.py
#  Created: 06/20/2018, 10:45
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines.utils import to_long_format, add_covariate_to_timeline

seed_df = pd.DataFrame.from_records([
    {'id': 'FB', 'E': True, 'T': 12, 'funding': 0},
    {'id': 'SU', 'E': True, 'T': 10, 'funding': 0},
])

cv = pd.DataFrame.from_records([
    {'id': 'FB', 'funding': 30, 't': 5},
    {'id': 'FB', 'funding': 15, 't': 10},
    {'id': 'FB', 'funding': 50, 't': 15},
    {'id': 'SU', 'funding': 10, 't': 6},
    {'id': 'SU', 'funding': 9,  't':  10},
])

# df = seed_df.pipe(to_long_format, 'T')\
#             .pipe(add_covariate_to_timeline, cv, 'id', 't', 'E', cumulative_sum=True)

df = seed_df.pipe(to_long_format, 'T')\
            .pipe(add_covariate_to_timeline, cv, 'id', 't', 'E', cumulative_sum=True)\
            .pipe(add_covariate_to_timeline, cv, 'id', 't', 'E', cumulative_sum=False)


"""
   start  cumsum_funding  funding  stop  id      E
0      0             0.0      0.0   5.0  FB  False
1      5            30.0     30.0  10.0  FB  False
2     10            45.0     15.0  12.0  FB   True
3      0             0.0      0.0   6.0  SU  False
4      6            10.0     10.0  10.0  SU  False
5     10            19.0      9.0  10.0  SU   True
"""
#==============================================================================
#==============================================================================
