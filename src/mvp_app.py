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

from predict_output import df, pred

# {'label':'Ebay', 'value':'c:...'}
dropdown_opts = [{'label':v, 'value':k} for k, v in df['name'].to_dict().items()]

app = dash.Dash()

app.layout = html.Div([
    html.Label('Exit Through the Gift Shop'),
    html.Div(
        dcc.Dropdown(
            id='input-list',
            options=dropdown_opts,
            value='eBay'
        )
    ),
    html.Button('Submit', id='button'),
    html.Div(id='output-container-button',
             children='Choose a value and press submit')
])

@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-list', 'value')])
def update_output(n_clicks, value): 
    # in_str = 'instagarage' # instagarage, ebay
    out_pred = pred.loc[value]
    out_info = df.loc[value]

    if out_pred.success:
        return "The company {} will likely exit successfully!".format(out_info['name'])
    else:
        return "The company {} will not, in all likelihood, exit successfully!".format(out_info['name'])

    # options=[
    #     {'label': 'New York City', 'value': 'NYC'},
    #     {'label': u'Montréal', 'value': 'MTL'},
    #     {'label': 'San Francisco', 'value': 'SF'}
    # ],

if __name__ == '__main__':
    app.run_server(debug=True)

#==============================================================================
#==============================================================================
