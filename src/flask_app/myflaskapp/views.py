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
# Take user input
from flask import request 

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.palettes import Inferno
import itertools

import json
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

# Import our model: 
# NOTE need to run "../run.py" from "flaskexample" in order to find 'a_Model'

# Load the data!
data_path = './data/'
df, Xan, pred, pred_probas, fp = pickle.load(open(data_path+'predictions.pkl', 'rb'))
fund_cv = pickle.load(open(data_path+'funding.pkl', 'rb'))
# List of company names for autocompletion
company_names = json.dumps(list(df.name.values))

def get_sim_idx(cid):
    """Get indices of most similar companies."""
    # cid = 'c:12' # get index of Twitter
    Xi = Xan.loc[cid].values.reshape(1, -1) # single sample
    C = cosine_similarity(Xi, Xan)
    C = pd.DataFrame(data=C, columns=Xan.index).abs().T # column vector
    idx = C.values.argsort(axis=0).squeeze()[::-1] # array shape (m,) descending
    sim_idx = C.iloc[idx[0:6]].index
    return sim_idx
    # df.loc[sim_idx]

def get_company_id(company_name):
    """Get company index from name."""
    try:
        return df[df['name'] == company_name].index[0]
    except IndexError as e:
        return render_template('500.html', error=e)

def color_gen():
    yield from itertools.cycle(Inferno[8])

def plot_timelines(sim_idx):
    """Plot funding timeline for each."""
    # create a new plot
    p = figure(tools='pan, box_zoom, reset, save', 
               x_axis_label='Time [years]', 
               y_axis_label='Funding Amt [$]')
    colors = color_gen()
    # TODO hover tips with series info + valuation
    for i, color in zip(range(len(sim_idx)), colors):
        s = sim_idx[i]
        the_co = fund_cv.loc[s].sort_values('funded_at')
        # Add [0,0] point
        time_to_funding = pd.Series(0, index=[the_co.index[0]])\
                            .append(the_co.time_to_funding) / 365
        raised_amount_usd = pd.Series(0, index=[the_co.index[0]])\
                            .append(the_co.raised_amount_usd)
        p.line(time_to_funding, raised_amount_usd,
               line_width=3, color=color, legend=df.loc[s, 'name'])
        p.circle(time_to_funding, raised_amount_usd, 
                 line_color=color, fill_color='white', size=8)
    return p

#------------------------------------------------------------------------------ 
#        Define the pages
#------------------------------------------------------------------------------
# The home page
@app.route('/')
def endgame_input():
    return render_template('index.html', company_names=company_names)

# The good stuff
@app.route('/output')
def endgame_output():
    """Make the output."""
    company_name = request.args.get('company_name')
    cid = get_company_id(company_name)

    # Get prediction class and probability
    pred_class = pred.loc[cid].idxmax() # [0, 1, 2, 3]
    if pred_class == 0:
        status = 'fail'
    elif pred_class == 1:
        status = 'exit in a timely fashion'
    elif pred_class == 2:
        status = 'exit, but in a long time'
    elif pred_class == 3:
        status = 'continue operating without exit'
    else:
        status = 'Error.'

    # Get list of similar companies
    sim_idx = get_sim_idx(cid)
    comps = df.loc[sim_idx[1:], ['name', 'category_code', 'founded_at', 'status']]

    # Make plot!
    plot = plot_timelines(sim_idx[::-1])
    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    # Embed plot into HTML via Flask Render
    script, div = components(plot)

    # TODO fp needs better names, descriptions (footnotes?)
    return render_template('output.html', company_name=company_name, fp=fp,
                           company_names=company_names,
                           status=status, comps=comps, plot_script=script,
                           plot_div=div, js_resources=js_resources,
                           css_resources=css_resources)

# Embed slides in app
@app.route('/about')
def endgame_about():
    return render_template('about.html')

#==============================================================================
#==============================================================================
