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

def sql2df(query, **kwargs):
    """Submit query to database and return dataframe with id as index."""
    # return pd.read_sql(query, con=engine, index_col='id', **kwargs)
    return pd.read_sql(query, con=engine, **kwargs)

#------------------------------------------------------------------------------ 
#        Initialize
#------------------------------------------------------------------------------
# TODAY = '2013/12/12' # date of snapshot YYYY/MM/DD
TODAY = '2014/01/10' # most recent founded_at date

# Initial DataFrame of ALL companies with at least one data point
query = ('SELECT o.id, ' +
         '       o.name, ' + 
         '       o.permalink, ' + 
         '       o.category_code, ' + 
         '       o.status, ' +
         '       o.founded_at ' +
         'FROM cb_objects AS o ' +
         "WHERE o.entity_type = 'company' AND o.founded_at IS NOT NULL")
df = sql2df(query, parse_dates=['founded_at'])

# NOTE status \in {'operating', 'acquired', 'ipo', 'closed'}

#------------------------------------------------------------------------------ 
#        Start the feature engineering!
#------------------------------------------------------------------------------
# 1. # employees -- can't get # employees at the date of exit...

# 2. # months company age (== (acq_date, or present) - (date of founding):
query = ('SELECT o.id, ' +
         '       MIN(a.acquired_at) AS acquir_at ' +
         'FROM cb_objects AS o ' +
         'JOIN cb_acquisitions AS a ' +
         '  ON o.id = a.acquired_object_id ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id')
rf = sql2df(query, parse_dates=['acquir_at'])
rf.columns = ['acquired_at'] # use better name
df = df.join(rf, on='id', how='left')
df['age_at_acq'] = df.acquired_at - df.founded_at

# 2b. # months company age at IPO: 
query = ('SELECT o.id, i.public_at ' +
         'FROM cb_objects AS o ' +
         'JOIN cb_ipos AS i ' +
         '  ON o.id = i.object_id ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query, parse_dates=['public_at'])
df = df.join(rf, on='id', how='left')
df['age_at_ipo'] = df.public_at - df.founded_at

# 2c. # months company age close: 
query = ('SELECT o.id, o.closed_at ' +
         'FROM cb_objects AS o ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query, parse_dates=['closed_at'])
df = df.join(rf, on='id', how='left')
df['age_at_close'] = df.closed_at - df.founded_at

# Consolidate ages into single column
df['age_at_exit'] = df[['age_at_acq', 'age_at_ipo', 'age_at_close']].min(axis=1)

# 3. # milestones on CrunchBase profile (i.e. Facebook added newsfeed): 
query = ('SELECT o.id, ' +
         '       (m.milestone_at < MIN(a.acquired_at)) AS is_before_acq ' +
         'FROM cb_objects AS o ' +
         'JOIN cb_milestones AS m ' +
         '  ON o.id = m.object_id ' +
         'JOIN cb_acquisitions AS a ' +
         '  ON o.id = a.acquired_object_id ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id, m.milestone_at')
rf = sql2df(query)
rf = rf.groupby('id').sum() # sum milestones completed
rf.columns = ['milestones']
df = df.join(rf, on='id', how='left')

# 8. Company Location
# query = ('SELECT o.id, o.region ' +
#          'FROM cb_objects AS o ' +
#          "WHERE o.entity_type = 'company'")
# rf = sql2df(query)
# df = df.join(rf, how='left')
# df.loc[df.region == 'unknown', 'region'] = np.nan

# 8b. Company Location -- (latitude, longitude)
query = ('SELECT o.id, l.latitude, l.longitude ' +
         'FROM cb_objects AS o ' +
         'JOIN cb_offices AS l ' +
         '  ON o.id = l.object_id ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
rf = rf.groupby('id').first() # remove duplicate IDs (
df = df.join(rf, on='id', how='left')

# 9. # offices: 
query = ('SELECT o.id, ' +
         '       COUNT(l.object_id) AS offices ' +
         'FROM cb_objects AS o  ' +
         'JOIN cb_offices AS l  ' +
         '  ON o.id = l.object_id  ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id')
rf = sql2df(query)
df = df.join(rf, on='id', how='left')

# 10. # products: 
query = ('SELECT o.id,  ' +
         '       COUNT(b.id) AS products  ' +
         'FROM cb_objects AS o  ' +
         'JOIN cb_objects AS b  ' +
         '  ON o.id = b.parent_id  ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id')
rf = sql2df(query)
df = df.join(rf, on='id', how='left')

# 12. # funding rounds: 
query = ('SELECT o.id, o.funding_rounds ' +
         'FROM cb_objects AS o ' + 
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, on='id', how='left')

# 13a. # investment rounds by the company: 
query = ('SELECT o.id, o.investment_rounds ' +
         'FROM cb_objects AS o ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, on='id', how='left')

# 13b. # companies invested in by the company: 
query = ('SELECT o.id, o.invested_companies ' +
         'FROM cb_objects AS o ' + 
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, on='id', how='left')

# 14. # acquisitions by the company:
query = ('SELECT o.id, ' +
         '       a1.acquired_at < MIN(a2.acquired_at) AS is_before_acq ' +
         'FROM cb_objects AS o  ' +
         'JOIN cb_acquisitions AS a1 ' +
         '  ON o.id = a1.acquiring_object_id  ' +
         'JOIN cb_acquisitions AS a2 ' +
         '  ON o.id = a2.acquired_object_id ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id, a1.acquired_at')
rf = sql2df(query)
rf = rf.groupby('id').sum(min_count=1) # sum milestones completed
rf.columns = ['acq_before_acq']
df = df.join(rf, on='id', how='left')

# 14b. # acquisitions by the company BEFORE IPO:
query = ('SELECT o.id, ' +
         '       SUM(a1.acquired_at < a2.public_at) AS acq_before_ipo ' +
         'FROM cb_objects AS o  ' +
         'JOIN cb_acquisitions AS a1 ' +
         '  ON o.id = a1.acquiring_object_id  ' +
         'JOIN cb_ipos AS a2 ' +
         '  ON a1.acquiring_object_id = a2.object_id ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id')
rf = sql2df(query)
df = df.join(rf, on='id', how='left')

# Consolidate acquisitions into single column
df['acq_before_exit'] = df[['acq_before_acq', 'acq_before_ipo']].min(axis=1)

# 15. # VC and PE firms investing in the company (total):
query = ('SELECT o.id,  ' +
         '       COUNT(i.investor_object_id) AS investors ' +
         'FROM cb_objects AS o  ' +
         'JOIN cb_funding_rounds AS f  ' +
         '  ON o.id = f.object_id  ' +
         'JOIN cb_investments AS i  ' +
         '  ON i.funding_round_id = f.funding_round_id  ' +
         'JOIN cb_objects AS b  ' +
         '  ON i.investor_object_id = b.id  ' +
         "WHERE b.entity_type = 'FinancialOrg' " +
         'GROUP BY o.id')
rf = sql2df(query)
df = df.join(rf, on='id', how='left')

# 18. # investors per round == # investors (15) / # funding rounds (13a)
df['investors_per_round'] = df.investors / df.funding_rounds

# 19. $ per round == cb_objects.funding_total_usd / # funding rounds (12)
query = ('SELECT o.id, o.funding_total_usd ' +
         'FROM cb_objects AS o ' + 
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, on='id', how='left')
df['funding_per_round'] = df.funding_total_usd / df.funding_rounds

# 19b. Time between rounds
query = """
SELECT o.id,
       TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) AS time_to_funding
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
WHERE f.raised_amount_usd IS NOT NULL
    AND TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) IS NOT NULL
"""
rf = sql2df(query)
# TODO Convert this line to ".diff().mean()"!!
# Otherwise we're just getting an analagous value to age_at_exit
# Probably need to NOT index by id since we get duplicates... just use integer
# indexing to guarantee unique value, and then join(..., on='id')
# Try this?? Need to get mean of diff...
# rf['diff'] = rf.sort_values('funded_at').groupby('id').time_to_funding.diff()
rf = rf.time_to_funding.groupby('id').mean()
rf.name = 'avg_time_to_funding'
rf = rf[rf >= 0] # eliminate negative numbers
df = df.join(rf, on='id', how='inner')

# Last funding round code
query = """
SELECT  o.id,
        f.funding_round_code
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
WHERE o.founded_at IS NOT NULL
    AND o.last_funding_at = f.funded_at
"""
rf = sql2df(query)
rf.columns = ['last_funding_round_code']
df = df.join(rf, on='id', how='inner')

# 20. founder experience
query = ('SELECT o1.id, ' +
         '       o2.id AS employee,         ' +
         '       MIN(r2.start_at) as earliest_start ' +
         'FROM cb_objects AS o1   ' +
         'JOIN cb_relationships AS r1   ' +
         '  ON o1.id = r1.relationship_object_id   ' +
         'JOIN cb_objects AS o2  ' +
         '  ON r1.person_object_id = o2.id  ' +
         'JOIN cb_relationships AS r2  ' +
         '  ON o2.id = r2.person_object_id  ' +
         "WHERE o1.entity_type = 'company' " +
         "      AND r1.title RLIKE 'founder|board|director'  " +
         'GROUP BY o1.id, o2.id')
rf = sql2df(query, parse_dates=['earliest_start'])
rf = rf.join(df.founded_at, on='id', how='left')
rf['experience'] = rf.founded_at - rf.earliest_start
rf.loc[rf.experience < pd.Timedelta(0), 'experience'] = pd.NaT
rf = rf.experience.dt.days.groupby('id').sum(min_count=1) # sum milestones completed
df = df.join(rf, on='id', how='left')

# Add funding round code as categorical values
dvs = pd.get_dummies(df.last_funding_round_code)
df = df.join(dvs, on='id', how='inner')

#------------------------------------------------------------------------------ 
#        Clean up
#------------------------------------------------------------------------------
# Need to fill some NaN values 
df.category_code.fillna(value='other', inplace=True)

# Fill NaT values to company age at present
df.age_at_exit.fillna(value=(pd.Timestamp(TODAY) - df.founded_at), inplace=True)
df.age_at_exit = df.age_at_exit.dt.days # convert to floats

# NOTE fillna with lat/lon of city

# Drop any rows where ALL relevant info is NaN
cols = ['acquired_at', 'age_at_acq', 'public_at', 'age_at_ipo', 'closed_at',
       'age_at_close', 'milestones', 'latitude', 'longitude',
       'offices', 'products', 'funding_rounds', 'investment_rounds',
       'invested_companies', 'acq_before_acq', 'acq_before_ipo',
       'acq_before_exit', 'investors', 'investors_per_round',
       'funding_total_usd', 'funding_per_round', 'experience']
df.dropna(axis=0, subset=cols, how='all', inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)
df = df[~df.index.duplicated(keep='first')]

#------------------------------------------------------------------------------ 
#        Create labels
#------------------------------------------------------------------------------
# What is a successful exit? acquisition or IPO BEFORE predefined time window
std_ages = df.groupby('category_code').std().age_at_exit
mean_ages = df.groupby('category_code').mean().age_at_exit
threshold = mean_ages + 2*std_ages

# Set age threshold for each label
df['threshold'] = np.nan
for label in threshold.index:
    df.loc[df.category_code == label, 'threshold'] = threshold[label]

# Multi-classification labels
df['label'] = np.nan
# Failure
df.loc[df.status == 'closed', 'label'] = 0
# Success
df.loc[(df.status == 'acquired') | (df.status == 'ipo'), 'label'] = 1
# Operating, not likely to exit
df.loc[(df.status == 'operating') 
        & (df.age_at_exit >= df.threshold), 'label'] = 2
# Operating, too early to tell?
df.loc[(df.status == 'operating') 
        & (df.age_at_exit < df.threshold), 'label'] = 3

# Write final dataframe to csv
# filename = '../data/cb_input_multi_idcol.pkl'
# print('Writing features to \'{}\'...'.format(filename))
# df.to_pickle(filename)
# print('done.')
#==============================================================================
#==============================================================================
