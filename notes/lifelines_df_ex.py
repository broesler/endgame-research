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

from lifelines.utils import covariates_from_event_matrix
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

# Event matrix
base_df = pd.DataFrame.from_records([{'id':1, 'E1':1, 'E3':2}, 
                                     {'id':2, 'E2':5},
                                     {'id':3, 'E1':3, 'E2':5, 'E3':7}])
"""
    id    E1      E2     E3
0   1     1.0     NaN    2.0
1   2     NaN     5.0    NaN
2   3     3.0     5.0    7.0
"""

cv = covariates_from_event_matrix(base_df, 'id')
"""
event  id  duration  E1  E2  E3
0       1       1.0   1   0   0
1       1       2.0   0   0   1
2       2       5.0   0   1   0
3       3       3.0   1   0   0
4       3       5.0   0   1   0
5       3       7.0   0   0   1
"""


#==============================================================================
#==============================================================================
