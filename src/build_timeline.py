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

from functools import reduce
import pickle
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
#        Static features (don't change after founding)
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

# founder experience
query = """
SELECT o1.id,
       o2.id AS employee,         
       MIN(r2.start_at) as earliest_start 
FROM cb_objects AS o1
JOIN cb_relationships AS r1
  ON o1.id = r1.relationship_object_id
JOIN cb_objects AS o2  
  ON r1.person_object_id = o2.id
JOIN cb_relationships AS r2  
  ON o2.id = r2.person_object_id  
WHERE o1.entity_type = 'company'
    AND r1.title RLIKE 'founder|board|director'
GROUP BY o1.id, o2.id
"""
rf = sql2df(query, parse_dates=['earliest_start'])
rf = rf.merge(df[['id', 'dates']], on='id', how='inner')
rf['experience'] = (rf.dates - rf.earliest_start).dt.days
rf.loc[rf.experience < 0, 'experience'] = np.nan
rf = rf.groupby('id', as_index=False).experience.sum()
df = df.merge(rf, on='id', how='inner')

#------------------------------------------------------------------------------ 
#        Build Timeline dataframe
#------------------------------------------------------------------------------
tf = df[['id', 'dates']].copy()
tf['event_id'] = 'founded'

# Funding rounds
query = """
SELECT o.id,
       f.funded_at AS dates,
       f.raised_amount_usd AS famt,
       f.participants AS investors
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
WHERE o.founded_at IS NOT NULL 
    AND f.funded_at IS NOT NULL
"""
# WHERE TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) IS NOT NULL
rf = sql2df(query, dates=True)
rf['event_id'] = 'funded'

# Append events to timeline (essentially an inner join with newest data)
tf = tf[tf.id.isin(rf.id)].append(rf, ignore_index=True)

tf['famt_cumsum'] = tf.loc[tf.event_id == 'funded']\
                      .sort_values('dates')\
                      .groupby('id')['famt'].cumsum()

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

# milestones on CrunchBase profile (i.e. Facebook added newsfeed): 
query = """
SELECT o.id,
       m.milestone_at AS dates
FROM cb_objects AS o 
JOIN cb_milestones AS m 
  ON o.id = m.object_id 
WHERE o.entity_type = 'company' 
"""
rf = sql2df(query, dates=True)
rf['event_id'] = 'milestone'
tf = append_to_timeline(tf, rf)

# investment rounds (in other companies) by the company,
# and total amount of funding in those rounds
query = """
SELECT o.id,
       f.funded_at AS dates,
       f.raised_amount_usd AS iamt,
       f.participants AS iparts
FROM cb_objects AS o
JOIN cb_investments AS i
    ON o.id = i.investor_object_id
JOIN cb_funding_rounds AS f
    ON i.funding_round_id = f.funding_round_id
WHERE f.funded_at IS NOT NULL
    AND o.entity_type = 'company'
"""
rf = sql2df(query, dates=True)
rf['event_id'] = 'investment'
tf = append_to_timeline(tf, rf)

# acquisitions by the company:
query = """
SELECT o.id,
       a.acquired_at AS dates,
       a.price_amount AS aamt
FROM cb_objects AS o  
JOIN cb_acquisitions AS a
  ON o.id = a.acquiring_object_id
WHERE o.entity_type = 'company' 
"""
rf = sql2df(query, dates=True)
rf['event_id'] = 'acquisition'
tf = append_to_timeline(tf, rf)

# 751 out of 3235 acquisitions have values attached.

# Limit to companies with founding dates (should have happened already??)
founded_ids = tf.loc[tf.event_id == 'founded'].id
tf = tf.loc[tf.id.isin(founded_ids)]

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

# Convert to floats
tf.time_diff = tf.time_diff.dt.days
tf.time_to_event = tf.time_to_event.dt.days

# NOTE this line may eliminate NaN values unintentionally:
# Get rid of negative times (<1% of rows)
tf = tf[(tf.time_diff >= 0) & (tf.time_to_event >= 0)]

# Test subset:
test_tf = tf[(tf.id == 'c:12') | (tf.id == 'c:126')].copy()
# test_g = test_tf.sort_values('dates').groupby(['id', 'event_id'])
# print(test_tf.sort_values(['id', 'time_to_event']))

#------------------------------------------------------------------------------ 
#        Clean up
#------------------------------------------------------------------------------
# Filter df to match indices of tf (save space!)
df = df[df.id.isin(tf.id)]

# Write final dataframe to csv
filename = '../data/cb_input_datasets_full.pkl'
print('Writing features to \'{}\'...'.format(filename))
pickle.dump([tf, df], open(filename, 'wb'))
print('done.')
#==============================================================================
#==============================================================================
