#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: timeseries_features.py
#  Created: 06/19/2018, 10:05
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, AalenAdditiveFitter
from lifelines.utils import k_fold_cross_validation
from lifelines.utils import to_long_format, add_covariate_to_timeline

from sqlalchemy import create_engine

df = pd.read_pickle('../data/cb_input.pkl')

# Open connection to the database
db_name = 'crunchbase_db'
engine = create_engine('mysql://bernie@localhost:5432/' + db_name)

def sql2df(query, **kwargs):
    """Submit query to database and return dataframe with id as index."""
    return pd.read_sql(query, con=engine, index_col='id', **kwargs)

# Possible rounds:
# mysql> SELECT DISTINCT funding_round_code FROM cb_funding_rounds;
# +--------------------+
# | funding_round_code |
# +--------------------+
# | b                  |
# | angel              |
# | a                  |
# | seed               |
# | c                  |
# | d                  |
# | unattributed       |
# | debt_round         |
# | e                  |
# | f                  |
# | private_equity     |
# | grant              |
# | post_ipo_equity    |
# | post_ipo_debt      |
# | partial            |
# | convertible        |
# | crowd              |
# | g                  |
# | secondary_market   |
# | crowd_equity       |
# +--------------------+
# 20 rows in set (0.03 sec)
#
# mysql> SELECT DISTINCT funding_round_type FROM cb_funding_rounds;
# +--------------------+
# | funding_round_type |
# +--------------------+
# | series-b           |
# | angel              |
# | series-a           |
# | series-c+          |
# | venture            |
# | other              |
# | private-equity     |
# | post-ipo           |
# | crowdfunding       |
# +--------------------+
# 9 rows in set (0.02 sec)

#------------------------------------------------------------------------------ 
#        Get funding round data
#------------------------------------------------------------------------------
query = """
SELECT  DISTINCT o.id,
        o.name,
        f.funding_round_type,
        f.funding_round_code,
        f.funded_at,
        f.raised_amount_usd,
        TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) AS time_to_funding
FROM cb_objects AS o 
JOIN cb_funding_rounds AS f 
ON o.id = f.object_id
WHERE f.raised_amount_usd IS NOT NULL
    AND TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) IS NOT NULL
"""
rf = sql2df(query, parse_dates=['funded_at'])

# Join static company data
df = df.join(rf.drop(columns='name'), how='inner')
df['id'] = df.index
df.reset_index(drop=True, inplace=True) # revert to integer indexing

# Create index label for each fund
df['fund_idx'] = df.sort_values('funded_at').groupby('id').cumcount()

dr = df[(df.name == 'Twitter') | (df.name == 'Facebook')]

# test = dr.pipe(to_long_format, 'T')\
#          .pipe(add_covariate_to_timeline, cv, 'id', 't', 'E', cumulative_sum=True)\
#          .pipe(add_covariate_to_timeline, cv, 'id', 't', 'E', cumulative_sum=False)


# df_t = rf[rf.name == 'Twitter']
# df_f = rf[rf.name == 'Facebook']
# fig = plt.figure(9)
# plt.clf()
# ax = plt.gca()
# plt.plot(df_t.time_to_funding, df_t.raised_amount_usd, '-xb', markersize=10)
# plt.plot(df_f.time_to_funding, df_f.raised_amount_usd, '-xr', markersize=10)
# # plt.plot(df_t.time_to_funding, df_t.pre_money_valuation, '-ob', markersize=10)
# # plt.plot(df_f.time_to_funding, df_f.pre_money_valuation, '-or', markersize=10)
# plt.plot(df_t.time_to_funding, df_t.post_money_valuation, '-sb', markersize=10)
# plt.plot(df_f.time_to_funding, df_f.post_money_valuation, '-sr', markersize=10)
# ax.set_yscale('log')

plt.show()

#==============================================================================
#==============================================================================
