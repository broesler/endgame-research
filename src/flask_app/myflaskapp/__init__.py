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
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

from myflaskapp import views

#==============================================================================
#==============================================================================
