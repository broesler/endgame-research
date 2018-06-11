#!/usr/local/bin/python3
#==============================================================================
#     File: make_odm_sql.py
#  Created: 06/08/2018, 23:59
#   Author: Bernie Roesler
#
"""
  Description: Convert ODM csv files into tables in crunchbase_db
"""
#==============================================================================

import pandas as pd
import numpy as np

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

db_name = 'crunchbase_db'

#------------------------------------------------------------------------------ 
#       Start the SQL engine/session
#------------------------------------------------------------------------------
engine = create_engine('mysql://bernie@localhost:5432/' + db_name,\
                        echo=False)

Session = sessionmaker(bind=engine) # create Session object
session = Session()                 # open a connection

#------------------------------------------------------------------------------ 
#        Load the csv files
#------------------------------------------------------------------------------
# test = pd.DataFrame(np.arange(9).reshape((3,3)), columns=['a', 'b', 'c'])

metadata = MetaData()

names = ['organizations', 'people']
# names = ['organizations']
for name in names:
    df = pd.read_csv('./odm_csv/' + name + '.csv')
    # df.to_sql(name, con=engine, index=False, if_exists='replace', chunksize=1000)

#==============================================================================
#==============================================================================
