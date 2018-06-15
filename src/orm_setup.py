#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: orm_setup.py
#  Created: 06/13/2018, 15:04
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

# See: 
# <http://docs.sqlalchemy.org/en/latest/orm/extensions/automap.html#basic-use>
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

Base = automap_base()

db_name = 'crunchbase_db'
engine = create_engine('mysql://bernie@localhost:5432/' + db_name)

# NOTE a more robust implementation of the queries below would rely on the ORM
# model in sqlalchemy. But time is money!!

# reflect the tables
Base.prepare(engine, reflect=True)

# mapped classes are now created with names by default
# matching that of the table name.
Acquisitions   = Base.classes.cb_acquisitions  
Degrees        = Base.classes.cb_degrees       
Funding_rounds = Base.classes.cb_funding_rounds
Funds          = Base.classes.cb_funds         
Investments    = Base.classes.cb_investments   
Ipos           = Base.classes.cb_ipos          
Milestones     = Base.classes.cb_milestones    
Objects        = Base.classes.cb_objects       
Offices        = Base.classes.cb_offices       
# People         = Base.classes.cb_people       # from snapshot .sql
Relationships  = Base.classes.cb_relationships 
# Organizations  = Base.classes.organizations     # broken!!
People         = Base.classes.people            # from odm/people.csv

# Start a session (for queries, etc.)
session = Session(engine)

Test query
q = session.query(People).first()
print(q.id, q.first_name, q.last_name)

#==============================================================================
#==============================================================================
