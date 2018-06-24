#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: build_features.py
#  Created: 06/11/2018, 21:16
#   Author: Bernie Roesler
#
"""
  Description: Build desired features into their own table.
"""
#==============================================================================

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

# Open connection to the database
db_name = 'crunchbase_db'
engine = create_engine('mysql://bernie@localhost:5432/' + db_name)

def sql2df(query, dates=False, **kwargs):
    """Submit query to database and return dataframe with id as index."""
    if dates:
        return pd.read_sql(query, con=engine, parse_dates=['dates'], **kwargs)
    else:
        return pd.read_sql(query, con=engine, **kwargs)

def append_to_timeline(tf, rf):
    """ Append frame on right using only ids of frame on left.

    Ensure that we don't add rows for events that did not occur for a given
    company.
    """
    return tf.append(rf.loc[rf.id.isin(tf.id) & rf.dates.notnull()],
                     ignore_index=True)

#------------------------------------------------------------------------------ 
#        Initialize
#------------------------------------------------------------------------------
# TODAY = '2013/12/12' # date of snapshot YYYY/MM/DD
TODAY = '2014/01/10' # most recent founded_at date

# Initial DataFrame of ALL companies with at least one data point
query = """
SELECT o.id,
       o.name,  
       o.category_code,  
       o.status,
       o.founded_at AS dates
FROM cb_objects AS o 
WHERE o.entity_type = 'company' 
    AND o.founded_at IS NOT NULL
"""
df = sql2df(query, dates=True)

# NOTE status \in {'operating', 'acquired', 'ipo', 'closed'}

#------------------------------------------------------------------------------ 
#        Get static features
#------------------------------------------------------------------------------
# Company Location -- (latitude, longitude)
query = """
SELECT o.id, l.latitude, l.longitude
FROM cb_objects AS o 
JOIN cb_offices AS l 
  ON o.id = l.object_id 
WHERE o.entity_type = 'company'
"""
rf = sql2df(query)
rf = rf.groupby('id').first().reset_index() # remove duplicate IDs (
df = df.merge(rf, on='id', how='left')

# offices: 
query = """
SELECT o.id,
       COUNT(l.object_id) AS offices 
FROM cb_objects AS o  
JOIN cb_offices AS l  
  ON o.id = l.object_id  
WHERE o.entity_type = 'company' 
GROUP BY o.id
"""
rf = sql2df(query)
df = df.merge(rf, on='id', how='left')

# products: 
query = """
SELECT o.id,
       COUNT(b.id) AS products  
FROM cb_objects AS o  
JOIN cb_objects AS b  
  ON o.id = b.parent_id  
WHERE o.entity_type = 'company' 
GROUP BY o.id
"""
rf = sql2df(query)
df = df.merge(rf, on='id', how='left')

#------------------------------------------------------------------------------ 
#        Build Timeline dataframe
#------------------------------------------------------------------------------
tf = df[['id', 'dates']].copy()
tf['event_id'] = 'founded'

# Funding rounds
query = """
SELECT o.id,
       f.funded_at AS dates
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
WHERE TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) IS NOT NULL
"""
rf = sql2df(query, dates=True)
rf['event_id'] = 'funded'

# Append events to timeline (essentially an inner join with newest data)
tf = tf[tf.id.isin(rf.id)].append(rf, ignore_index=True)

# company age at acquisition
query = """
SELECT o.id,
       MIN(a.acquired_at) AS dates
FROM cb_objects AS o 
JOIN cb_acquisitions AS a 
  ON o.id = a.acquired_object_id 
WHERE o.entity_type = 'company' 
GROUP BY o.id
"""
rf = sql2df(query, dates=True)
rf['event_id'] = 'acquired'
tf = append_to_timeline(tf, rf)

# company age at IPO: 
query = """
SELECT o.id, 
       i.public_at AS dates
FROM cb_objects AS o 
JOIN cb_ipos AS i 
  ON o.id = i.object_id 
WHERE o.entity_type = 'company'
"""
rf = sql2df(query, dates=True)
rf['event_id'] = 'public'
tf = append_to_timeline(tf, rf)

# company age at close: 
query = """
SELECT o.id, 
       o.closed_at AS dates
FROM cb_objects AS o 
WHERE o.entity_type = 'company'
"""
rf = sql2df(query, dates=True)
rf['event_id'] = 'closed'
tf = append_to_timeline(tf, rf)

#------------------------------------------------------------------------------ 
#        Add timing features for all events
#------------------------------------------------------------------------------
# Get time_to_event for each group
def get_time_to(group):
    """Get time to event from founding date given a group."""
    founded_at = group.loc[group.event_id == 'founded'].dates.values[0]
    group['time_to_event'] = group.dates - founded_at
    return group
tf = tf.groupby('id').apply(get_time_to)

# Event count and time between events
g = tf.sort_values('dates').groupby(['id', 'event_id'])
tf['event_count'] = g.cumcount() + 1

# Calculate time between events
tf['time_diff'] = g.time_to_event.diff()
# Replace diff NaN with initial time_to_funding value (diff with 0)
tf.loc[tf.time_diff.isnull(), 'time_diff'] = tf.time_to_event

# NOTE this line may eliminate NaN values unintentionally:
# Get rid of negative times (<1% of rows)
tf = tf[(tf.time_diff >= pd.Timedelta(0)) 
        & (tf.time_to_event >= pd.Timedelta(0))]

# Test subset:
test_tf = tf[(tf.id == 'c:12') | (tf.id == 'c:126')].copy()
# test_g = test_tf.sort_values('dates').groupby(['id', 'event_id'])
# print(test_tf.sort_values(['id', 'time_to_event']))

# Reset index to (id, integer)
# test_tf = test_tf.sort_values(['id', 'time_to_event'])\
#                  .set_index(['id', test_tf.index])

#------------------------------------------------------------------------------ 
#        Other features to add
#------------------------------------------------------------------------------
# # milestones on CrunchBase profile (i.e. Facebook added newsfeed): 
# query = """
# SELECT o.id,
#        m.milestone_at AS dates
# FROM cb_objects AS o 
# JOIN cb_milestones AS m 
#   ON o.id = m.object_id 
# WHERE o.entity_type = 'company' 
# """
# rf = sql2df(query)
# rf['milestones'] = rf.sort_values('dates').groupby('id').cumcount()
# df = df.merge(rf, on='id', how='left')

# # investment rounds (in other companies) by the company,
# # and total amount of funding in those rounds
# query = """
# SELECT o.id,
#        f.funded_at AS invested_at,
#        f.raised_amount_usd AS iamt,
#        f.participants AS iparts
# FROM cb_objects AS o
# JOIN cb_investments AS i
#     ON o.id = i.investor_object_id
# JOIN cb_funding_rounds AS f
#     ON i.funding_round_id = f.funding_round_id
# WHERE f.funded_at IS NOT NULL
#     AND o.entity_type = 'company'
# """
# rf = sql2df(query)
# rf['investments'] = rf.sort_values('invested_at').groupby('id').cumcount()
# rf['tot_iamt'] = rf.sort_values('invested_at').groupby('id').iamt.cumsum()
# # TODO include # co-investors per round (subtract one before cumsum()
# # rf['tot_coinvestors'] = rf.sort_values('invested_at').groupby('id').iparts.cumsum()
# df = df.merge(rf, on='id', how='left')
#
# # acquisitions by the company:
# query = """
# SELECT o.id,
#        a.acquired_at,
#        a.price_amount
# FROM cb_objects AS o  
# JOIN cb_acquisitions AS a
#   ON o.id = a.acquiring_object_id
# WHERE o.entity_type = 'company' 
# """
# rf = sql2df(query)
# rf['acquisitions'] = rf.sort_values('acquired_at').groupby('id').cumcount()
# rf['tot_aamt'] = rf.sort_values('acquired_at').groupby('id').price_amount.cumcount()
# rf.rename({'acquired_at':'made_acq_at'}, inplace=True)
# df = df.merge(rf, on='id', how='left')
#
# # 15. # VC and PE firms investing in the company (total):
# query = """
# SELECT o.id,
#        COUNT(i.investor_object_id) AS investors 
# FROM cb_objects AS o  
# JOIN cb_funding_rounds AS f  
#   ON o.id = f.object_id  
# JOIN cb_investments AS i  
#   ON i.funding_round_id = f.funding_round_id  
# JOIN cb_objects AS b  
#   ON i.investor_object_id = b.id  
# WHERE b.entity_type = 'FinancialOrg' 
# GROUP BY o.id
# """
# rf = sql2df(query)
# df = df.merge(rf, on='id', how='left')
#
# # 18. # investors per round == # investors (15) / # funding rounds (13a)
# df['investors_per_round'] = df.investors / df.funding_rounds
#
# # Last funding round code
# query = """
# SELECT  o.id,
#         f.funding_round_code
# FROM cb_objects AS o 
# JOIN cb_funding_rounds AS f 
# ON o.id = f.object_id
# WHERE o.founded_at IS NOT NULL
#     AND o.last_funding_at = f.funded_at
# """
# rf = sql2df(query)
# rf.columns = ['id', 'last_funding_round_code']
# df = df.merge(rf, on='id', how='left')
#
# # 20. founder experience
# query = """
# SELECT o1.id,
#        o2.id AS employee,         
#        MIN(r2.start_at) as earliest_start 
# FROM cb_objects AS o1
# JOIN cb_relationships AS r1
#   ON o1.id = r1.relationship_object_id
# JOIN cb_objects AS o2  
#   ON r1.person_object_id = o2.id
# JOIN cb_relationships AS r2  
#   ON o2.id = r2.person_object_id  
# WHERE o1.entity_type = 'company'
#     AND r1.title RLIKE 'founder|board|director'
# GROUP BY o1.id, o2.id
# """
# rf = sql2df(query, parse_dates=['earliest_start'])
# rf = rf.merge(df.founded_at, on='id', how='left')
# rf['experience'] = rf.founded_at - rf.earliest_start
# rf.loc[rf.experience < pd.Timedelta(0), 'experience'] = pd.NaT
# rf = rf.experience.dt.days.groupby('id').sum(min_count=1).reset_index()
# df = df.merge(rf, on='id', how='left')
#
# # Add funding round code as categorical values
# dvs = pd.get_dummies(df.last_funding_round_code)
# df = df.merge(dvs, on='id', how='left')
#
# #------------------------------------------------------------------------------ 
# #        Clean up
# #------------------------------------------------------------------------------
# # Need to fill some NaN values 
# df.category_code.fillna(value='other', inplace=True)
#
# # Fill NaT values to company age at present
# df.age_at_exit.fillna(value=(pd.Timestamp(TODAY) - df.founded_at), inplace=True)
# df.age_at_exit = df.age_at_exit.dt.days # convert to floats
#
# # NOTE fillna with lat/lon of city
#
# # Drop any rows where ALL relevant info is NaN
# cols = ['acquired_at', 'age_at_acq', 'public_at', 'age_at_ipo', 'closed_at',
#        'age_at_close', 'milestones', 'latitude', 'longitude',
#        'offices', 'products', 'funding_rounds', 'investment_rounds',
#        'invested_companies', 'acq_before_acq', 'acq_before_ipo',
#        'acq_before_exit', 'investors', 'investors_per_round',
#        'funding_total_usd', 'funding_per_round', 'experience']
# df.dropna(axis=0, subset=cols, how='all', inplace=True)
#
# # Drop duplicates
# df.drop_duplicates(inplace=True)
# df = df[~df.index.duplicated(keep='first')]
#
# #------------------------------------------------------------------------------ 
# #        Create labels
# #------------------------------------------------------------------------------
# # What is a successful exit? acquisition or IPO BEFORE predefined time window
# std_ages = df.groupby('category_code').std().age_at_exit
# mean_ages = df.groupby('category_code').mean().age_at_exit
# threshold = mean_ages + 2*std_ages
#
# # Set age threshold for each label
# df['threshold'] = np.nan
# for label in threshold.index:
#     df.loc[df.category_code == label, 'threshold'] = threshold[label]
#
# # Multi-classification labels
# df['label'] = np.nan
# # Failure
# df.loc[df.status == 'closed', 'label'] = 0
# # Success
# df.loc[(df.status == 'acquired') | (df.status == 'ipo'), 'label'] = 1
# # Operating, not likely to exit
# df.loc[(df.status == 'operating') 
#         & (df.age_at_exit >= df.threshold), 'label'] = 2
# # Operating, too early to tell?
# df.loc[(df.status == 'operating') 
#         & (df.age_at_exit < df.threshold), 'label'] = 3
#
# # Write final dataframe to csv
# filename = '../data/cb_input_multi_idcol.pkl'
# print('Writing features to \'{}\'...'.format(filename))
# df.to_pickle(filename)
# print('done.')
# #==============================================================================
# #==============================================================================
