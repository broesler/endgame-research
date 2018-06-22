#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: views.py
#  Created: 06/18/2018, 20:42
#   Author: Bernie Roesler
#
"""
  Description: Define views for the site.
"""
#==============================================================================

from myflaskapp import app
from flask import render_template
import pandas as pd

# Take user input
from flask import request 

# Import our model: 
# NOTE need to run "../run.py" from "flaskexample" in order to find 'a_Model'
from a_Model import ModelIt
# More "robust" way to include more modules with individual directories, still
# see above note.
# from a_Model_test.a_Model import ModelIt

user = 'bernie' #add your username here (same as previous postgreSQL)                      
host = 'localhost'

# The home page
@app.route('/')
def endgame_input():
    return render_template("index.html")

@app.route('/output')
# Dummy out function:
def endgame_output():
    return render_template("output.html")

#==============================================================================
#==============================================================================
