#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: build_features.py
#  Created: 06/24/2018, 18:05
#   Author: Bernie Roesler
#
"""
  Description: Auxiliary functions to build features from given time slice
"""
#==============================================================================

import pandas as pd
import numpy as np

def make_labels(tf, df):
    """Label company outcomes given timeline and static dataframe."""
    y = df[['id']].copy()
    y['label'] = np.nan

    # Failures
    closures = tf[tf.event_id == 'closed']
    y.loc[y.id.isin(closures.id), 'label'] = 0

    # What is a successful exit? 
    # "Exit" == acquisition or IPO at least before 1-std beyond mean age of
    # acquisition for a given industry
    # NOTE workaround: groupby().std() does NOT work on timeseries64, but
    # describe() does!?
    exits = tf[(tf.event_id == 'public') | (tf.event_id == 'acquired')]\
            .merge(df, on='id', how='inner')
    g = exits.groupby('category_code')
    threshold = g.mean().time_to_event + g.std().time_to_event # for ALL acquisitions

    # Set age threshold for each label
    exits['threshold'] = np.nan
    for label in threshold.index:
        exits.loc[exits.category_code == label, 'threshold'] = threshold[label]

    # Get id of companies that have exited
    timely_exits = exits[exits.time_to_event < exits.threshold]
    y.loc[y.id.isin(timely_exits.id), 'label'] = 1

    slow_exits = exits[exits.time_to_event >= exits.threshold]
    y.loc[y.id.isin(slow_exits.id), 'label'] = 2

    y.fillna(value=3, inplace=True)
    return y

# # Multi-classification labels
# tf['label'] = np.nan
# # Failure
# tf.loc[tf.status == 'closed', 'label'] = 0
# # Success
# tf.loc[(tf.status == 'acquired') | (tf.status == 'ipo'), 'label'] = 1
# # Operating, not likely to exit
# tf.loc[(tf.status == 'operating') 
#         & (tf.age_at_exit >= tf.threshold), 'label'] = 2
# # Operating, too early to tell?
# tf.loc[(tf.status == 'operating') 
#         & (tf.age_at_exit < tf.threshold), 'label'] = 3

#==============================================================================
#==============================================================================
