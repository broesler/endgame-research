#!/usr/local/bin/python3
#==============================================================================
#     File: starter_app.py
#  Created: 06/08/2018, 14:38
#   Author: Bernie Roesler
#
"""
  Description: Dash developer setup.
"""
#==============================================================================

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
            }
        ),

    html.Div(children='Dash: A web application framework for Python.', 
        style={
            'textAlign': 'center',
            'color': colors['text']
            }),

    dcc.Graph(
        id='example-graph',
        figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text'] 
                        }
                }
        }
    )
])

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.run_server(debug=True)

#==============================================================================
#==============================================================================
