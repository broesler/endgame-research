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
    return pd.read_sql(query, con=engine, index_col='id', **kwargs)

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
         "WHERE o.entity_type = 'company'")
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
df = df.join(rf, how='outer')
df['age_at_acq'] = df.acquired_at - df.founded_at

# 2b. # months company age at IPO: 
query = ('SELECT o.id, i.public_at ' +
         'FROM cb_objects AS o ' +
         'JOIN cb_ipos AS i ' +
         '  ON o.id = i.object_id ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query, parse_dates=['public_at'])
df = df.join(rf, how='outer')
df['age_at_ipo'] = df.public_at - df.founded_at

# 2c. # months company age close: 
query = ('SELECT o.id, o.closed_at ' +
         'FROM cb_objects AS o ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query, parse_dates=['closed_at'])
df = df.join(rf, how='outer')
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
df = df.join(rf, how='outer')

# 8. Company Location
# query = ('SELECT o.id, o.region ' +
#          'FROM cb_objects AS o ' +
#          "WHERE o.entity_type = 'company'")
# rf = sql2df(query)
# df = df.join(rf, how='outer')
# df.loc[df.region == 'unknown', 'region'] = np.nan

# 8b. Company Location -- (latitude, longitude)
query = ('SELECT o.id, l.latitude, l.longitude ' +
         'FROM cb_objects AS o ' +
         'JOIN cb_offices AS l ' +
         '  ON o.id = l.object_id ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
rf = rf.groupby('id').first() # remove duplicate IDs (
df = df.join(rf, how='outer')

# 9. # offices: 
query = ('SELECT o.id, ' +
         '       COUNT(l.object_id) AS offices ' +
         'FROM cb_objects AS o  ' +
         'JOIN cb_offices AS l  ' +
         '  ON o.id = l.object_id  ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id')
rf = sql2df(query)
df = df.join(rf, how='outer')

# 10. # products: 
query = ('SELECT o.id,  ' +
         '       COUNT(b.id) AS products  ' +
         'FROM cb_objects AS o  ' +
         'JOIN cb_objects AS b  ' +
         '  ON o.id = b.parent_id  ' +
         "WHERE o.entity_type = 'company' " +
         'GROUP BY o.id')
rf = sql2df(query)
df = df.join(rf, how='outer')

# 12. # funding rounds: 
query = ('SELECT o.id, o.funding_rounds ' +
         'FROM cb_objects AS o ' + 
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, how='outer')

# 13a. # investment rounds by the company: 
query = ('SELECT o.id, o.investment_rounds ' +
         'FROM cb_objects AS o ' +
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, how='outer')

# 13b. # companies invested in by the company: 
query = ('SELECT o.id, o.invested_companies ' +
         'FROM cb_objects AS o ' + 
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, how='outer')

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
df = df.join(rf, how='outer')

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
df = df.join(rf, how='outer')

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
df = df.join(rf, how='outer')

# 18. # investors per round == # investors (15) / # rounds (13a)
df['investors_per_round'] = df.investors / df.investment_rounds

# 19. $ per round == cb_objects.funding_total_usd / # funding rounds (12)
query = ('SELECT o.id, o.funding_total_usd ' +
         'FROM cb_objects AS o ' + 
         "WHERE o.entity_type = 'company'")
rf = sql2df(query)
df = df.join(rf, how='outer')
df['funding_per_round'] = df.funding_total_usd / df.funding_rounds

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
rf = rf.join(df.founded_at, how='left')
rf['experience'] = rf.founded_at - rf.earliest_start
rf.loc[rf.experience < pd.Timedelta(0), 'experience'] = pd.NaT
rf = rf.experience.dt.days.groupby('id').sum(min_count=1) # sum milestones completed
df = df.join(rf, how='outer')

# Clean up before labeling
Ntot = df.shape[0]
df.drop_duplicates(inplace=True)

# Need to fill NaN values 
df.category_code.fillna(value='other', inplace=True)
# df.age_at_exit.fillna(value=df.age_at_exit.max(), inplace=True)
df.age_at_exit.fillna(value=(pd.Timestamp(TODAY) - df.founded_at), inplace=True)

# NOTE fillna with lat/lon of city

# Fill NaT values to company age at present
df.age_at_exit = df.age_at_exit.dt.days # convert to floats

#------------------------------------------------------------------------------ 
#        Create labels
#------------------------------------------------------------------------------
# What is a successful exit? acquisition or IPO BEFORE predefined time window
# NOTE look at distribution of company age at exit before setting threshold
n_years = 6
threshold = n_years * 365

# One-hot encode output for random forest
df['success'] = 0
df['failure'] = 0
df.loc[df.status == 'acquired', 'success'] = 1 # Positive Examples
df.loc[df.status == 'ipo', 'success'] = 1
df.loc[df.status == 'closed', 'failure'] = 1 # Obvious negative examples
# Less obvious negative examples
df.loc[(df.status == 'operating') & (df.age_at_exit > threshold), 'failure'] = 1
 # Third class of pending
# df.loc[(df.status == 'operating') & (df.age_at_exit < threshold),
#        ['success', 'failure']] = [0, 0]

# Also create labeled output 
df['bin_label'] = 0
df.loc[(df.success == 1) & (df.failure == 0), 'bin_label'] = 1
df.loc[(df.success == 0) & (df.failure == 1), 'bin_label'] = 0
df.loc[(df.success == 0) & (df.failure == 0), 'bin_label'] = 0

df['mul_label'] = 0
df.loc[(df.success == 1) & (df.failure == 0), 'mul_label'] = 1
df.loc[(df.success == 0) & (df.failure == 1), 'mul_label'] = 0
df.loc[(df.success == 0) & (df.failure == 0), 'mul_label'] = 2

# Write final dataframe to csv
filename = '../data/cb_input.csv'
print('Writing features to \'{}\'...'.format(filename))
df.to_csv(filename)
print('done.')
#==============================================================================
#==============================================================================
