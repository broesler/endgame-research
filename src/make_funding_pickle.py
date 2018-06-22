#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: make_funding_pickle.py
#  Created: 06/21/2018, 20:14
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

import pandas as pd

from sqlalchemy import create_engine

# Open connection to the database
db_name = 'crunchbase_db'
engine = create_engine('mysql://bernie@localhost:5432/' + db_name)

# Get funding round data
query = """
SELECT o.id,
       o.name,
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
fund_cv['fund_idx'] = fund_cv.sort_values('funded_at').groupby('id').cumcount()
fund_cv.index = fund_cv.id
fund_cv.drop(columns='id', inplace=True)
fund_cv.to_pickle('../data/funding.pkl')

# df_t = fund_cv[fund_cv.name == 'Twitter']
# df_f = fund_cv[fund_cv.name == 'Facebook']
# fig = plt.figure(9)
# plt.clf()
# ax = plt.gca()
# plt.plot(df_t.time_to_funding, df_t.raised_amount_usd, '-xb', markersize=10)
# plt.plot(df_f.time_to_funding, df_f.raised_amount_usd, '-xr', markersize=10)
# ax.set_yscale('log')
# plt.tight_layout()
# plt.show()

#==============================================================================
#==============================================================================
