#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: core_components_app.py
#  Created: 06/14/2018, 16:32
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div([
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': u'Montréal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='MTL'
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)

#==============================================================================
#==============================================================================
