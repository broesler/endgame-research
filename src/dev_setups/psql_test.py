#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: psql_test.py
#  Created: 06/18/2018, 21:03
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================
## Python packages - you may have to pip install sqlalchemy, sqlalchemy_utils, and psycopg2.
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd

#In Python: Define your username and password used above. I've defined the database name (we're 
#using a dataset on births, so I call it birth_db). 
dbname = 'birth_db'
username = 'bernie'
pswd = ''

## 'engine' is a connection to a database
## Here, we're using postgres, but sqlalchemy can connect to other things too.
engine = create_engine('postgresql://%s:%s@localhost/%s' % (username, pswd, dbname))
# print('postgresql://%s:%s@localhost/%s' % (username, pswd, dbname))
# print(engine.url)
# # Replace localhost with IP address if accessing a remote server
#
# ## create a database (if it doesn't exist)
# if not database_exists(engine.url):
#     create_database(engine.url)
# print(database_exists(engine.url))
# print(engine.url)
#
# # load a database from the included CSV
# # edit the string below to account for where you saved the csv.
# csv_path = './births2012_downsampled.csv'
# birth_data = pd.read_csv(csv_path)
#
# ## insert data into database from Python (proof of concept - this won't be useful for big data, of course)
# ## df is any pandas dataframe 
# birth_data.to_sql('birth_data_table', engine, if_exists='replace')

## Now try the same queries, but in python!

# connect:
con = None
con = psycopg2.connect(database=dbname, user=username, host='localhost', password=pswd)
 
# query:
sql_query = """
SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';
"""
birth_data_from_sql = pd.read_sql_query(sql_query, con)

print(birth_data_from_sql.head())

#==============================================================================
#==============================================================================
