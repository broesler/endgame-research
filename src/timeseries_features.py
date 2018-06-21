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

from lifelines import CoxPHFitter, CoxTimeVaryingFitter, AalenAdditiveFitter
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import k_fold_cross_validation
from lifelines.utils import to_long_format, add_covariate_to_timeline

from sqlalchemy import create_engine

TODAY = '2014/01/10' # most recent founded_at date
base_df = pd.read_pickle('../data/cb_input.pkl')

# Open connection to the database
db_name = 'crunchbase_db'
engine = create_engine('mysql://bernie@localhost:5432/' + db_name)

#------------------------------------------------------------------------------ 
#        Get funding round data
#------------------------------------------------------------------------------
query = """
SELECT o.id,
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
fund_cv = pd.read_sql(query, con=engine, parse_dates=['funded_at'])

# Create index label for each fund -- BEFORE dummy vars
# fund_cv['fund_idx'] = fund_cv.sort_values('funded_at').groupby('id').cumcount()

# Convert funding_round_code to dummy variable
dvs = pd.get_dummies(fund_cv.funding_round_code)
fund_cv = fund_cv.join(dvs, how='inner')

# TODO Get acquisition data (same format as funding data with dates)
# query = """
# SELECT  DISTINCT o.id,
#         o.name,
#         f.funding_round_type,
#         f.funding_round_code,
#         f.funded_at,
#         f.raised_amount_usd,
#         TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) AS time_to_funding
# FROM cb_objects AS o 
# JOIN cb_funding_rounds AS f 
# ON o.id = f.object_id
# WHERE f.raised_amount_usd IS NOT NULL
#     AND TIMESTAMPDIFF(DAY, o.founded_at, f.funded_at) IS NOT NULL
# """
# fund_cv = pd.read_sql(query, con=engine, parse_dates=['funded_at'])

# TODO Get milestone data
# TODO Get relationship data?? when "key" employees brought on

# static_cols = ['name', 'category_code', 'founded_at',
#                'latitude', 'longitude', 'offices', 'products', 'experience', 
#                'success', 'failure', 'age_at_exit']
static_cols = ['name', 'category_code', 'success', 'age_at_exit']
# dynamic_cols = ['milestones', 'investment_rounds', 'invested_companies', 'investors']

df = base_df[static_cols]
df['id'] = df.index
df.reset_index(drop=True, inplace=True) # revert to integer indexing

# Merge fund_cv DataFrame
# df = df.merge(fund_cv.drop(columns='name'), on='id', how='inner')
# Reduce to just companies for which we have relevant information
df = df[df.id.isin(fund_cv.id.unique())]

# SMALL SUBSET TEST CODE:
# dr = df[(df.name == 'Twitter') | (df.name == 'Facebook')]
# lr = to_long_format(dr, duration_col='age_at_exit')
# rr = fund_cv.loc[(fund_cv.name == 'Twitter') | (fund_cv.name == 'Facebook'),
#             ['id', 'raised_amount_usd', 'time_to_funding']]
# lr = add_covariate_to_timeline(lr, rr, 'id', 'time_to_funding', 'success', cumulative_sum=True)

#------------------------------------------------------------------------------ 
#        Prepare DataFrame for lifelines analysis
#------------------------------------------------------------------------------
# lf = to_long_format(df, 'age_at_exit')
#
# # Piped version (possibly faster?)
# # lf = df.pipe(to_long_format, 'age_at_exit')\
# #        .pipe(add_covariate_to_timeline(fund_cv_amt, 
# #                                        'id', 'time_to_funding', 'success',
# #                                        cumulative_sum=False)\
#
# # NOTE WARNING THIS LINE IS SUPER FUCKING SLOW.
# # Add raised_amount_usd as time-varying covariate
# fund_cv_amt = fund_cv[['id', 'raised_amount_usd', 'time_to_funding']]
# lf = add_covariate_to_timeline(lf, fund_cv_amt, 'id', 'time_to_funding', 'success',
#                                cumulative_sum=False)
#
# # Add cumulative funding as covariate
# # lf = add_covariate_to_timeline(lf, fund_cv_amt, 
# #                                'id', 'time_to_funding', 'success',
# #                                cumulative_sum=True)
# # Add funding round type as covariate
#
# # WRITE TO PICKLE FILE!!!
# # lf.to_pickle('../data/survival_input.pkl')
#
# # df_t = fund_cv[fund_cv.name == 'Twitter']
# # df_f = fund_cv[fund_cv.name == 'Facebook']
# # fig = plt.figure(9)
# # plt.clf()
# # ax = plt.gca()
# # plt.plot(df_t.time_to_funding, df_t.raised_amount_usd, '-xb', markersize=10)
# # plt.plot(df_f.time_to_funding, df_f.raised_amount_usd, '-xr', markersize=10)
# # ax.set_yscale('log')
# # plt.tight_layout()
# # plt.show()
#
# #==============================================================================
# #==============================================================================
