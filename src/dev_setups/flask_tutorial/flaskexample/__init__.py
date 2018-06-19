#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: __init__.py
#  Created: 06/18/2018, 20:41
#   Author: Bernie Roesler
#
"""
  Description: Flask initialization file
"""
#==============================================================================

from flask import Flask

app = Flask(__name__)

from flaskexample import views

#==============================================================================
#==============================================================================
