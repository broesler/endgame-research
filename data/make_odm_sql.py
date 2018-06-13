#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: make_odm_sql.py
#  Created: 06/08/2018, 23:59
#   Author: Bernie Roesler
#
"""
  Description: Convert ODM csv files into tables in crunchbase_db.
"""
#==============================================================================

import pandas as pd
# import numpy as np

from sqlalchemy import create_engine

#------------------------------------------------------------------------------ 
#       Start the SQL engine/session
#------------------------------------------------------------------------------
db_name = 'crunchbase_db'
engine = create_engine('mysql://bernie@localhost:5432/' + db_name,\
                        echo=False)

#------------------------------------------------------------------------------ 
#        Load the csv files
#------------------------------------------------------------------------------
# test = pd.DataFrame(np.arange(9).reshape((3,3)), columns=['a', 'b', 'c'])

# names = ['organizations', 'people'] # 'people' works!
names = ['organizations']
for name in names:
    df = pd.read_csv('./odm_csv/' + name + '.csv')
    df.to_sql(name, con=engine, index=True, if_exists='replace', chunksize=5000)

# df_fund = pd.read_sql('cb_funding_rounds', con=engine)

#==============================================================================
#==============================================================================
