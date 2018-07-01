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

# Load the data!
data_path = './data/'
# pred, sim_idx, feat_imp, df, tf_fund, y = pickle.load(open(data_path + 'flask_db_full.pkl', 'rb'))
pred, sim_idx, _, tf_fund, y = pickle.load(open(data_path + 'flask_db_full.pkl', 'rb'))
tf, df = pickle.load(open(data_path + 'cb_input_datasets_full.pkl', 'rb'))
# List of company names for autocompletion (limit to startups!!)
company_names = json.dumps(list(df.loc[df.id.isin(pred.index), 'name'].values))

# TODO convert df.category to "display" names 

# NOTE won't need these lines with new database
pred[['proba0', 'proba1', 'proba2']] = pd.DataFrame(pred.probas.values.tolist(), index=pred.index)
pred.drop(columns='probas', inplace=True)

# Date of last funding event
tf_last_fund = tf_fund.groupby(['id'], as_index=False).max()

# i_start = 0     # Index of list to start on (init to top)
# N_display = 10  # Number of startups to display on homepage
# N_similar = 6   # Number of historical companies to show for comparison

def get_company_id(company_name):
    """Get company index from name."""
    try:
        return df.loc[df.name == company_name].id.values[0]
    except IndexError as e:
        return render_template('500.html', error=e)

def color_gen():
    yield from itertools.cycle(Inferno[8])

def plot_timelines(si):
    """Plot funding timeline for each given sim_idx[cid]."""
    # create a new plot
    p = figure(tools='pan, box_zoom, reset, save',
               x_axis_label='Time [years]',
               y_axis_label='Funding Amt [$]')
    colors = color_gen()
    # for i, color in zip(range(len(si)), colors):
        # TODO ERROR HANDLING!!! Not all companies have funding amounts!!
        # fillna values??
        # s = si[i]
        # Get cid from df
        # cid = df.loc[s].id
        # the_co = tf_fund.loc[tf_fund.id == cid].sort_values('dates')
        # # Add [0,0] point
        # time_to_funding = pd.Series(0, index=[the_co.index[0]])\
        #                     .append(the_co.time_to_event) / 365
        # raised_amount_usd = pd.Series(0, index=[the_co.index[0]])\
        #                     .append(the_co.famt_cumsum)
        # p.line(time_to_funding, raised_amount_usd,
        #        line_width=3, color=color, legend=df.loc[s, 'name'])
        # p.circle(time_to_funding, raised_amount_usd,
        #          line_color=color, fill_color='white', size=8)
    return p

#------------------------------------------------------------------------------
#        Define the pages
#------------------------------------------------------------------------------
# The home page
@app.route('/')
@app.route('/index')
def endgame_input():
    # Sort list of companies predicted to exit quickly by probability
    exits = pred.loc[pred.pred == 0].copy()
    exits.sort_values('proba0', ascending=False, inplace=True)
    top_exit_ids = exits.iloc[app.config['I_START']:app.config['N_DISPLAY']].index
    # Get general info
    top_exits = df.loc[df.id.isin(top_exit_ids)].copy()
    top_exits.rename(columns={'dates':'founding_date'}, inplace=True)
    # Get funding info
    top_exits = top_exits.merge(tf_last_fund.loc[tf_last_fund.id.isin(top_exit_ids)], on='id')
    top_exits.rename(columns={'dates':'last_funding_date'}, inplace=True)
    return render_template('index.html', title='Home', top_exits=top_exits)

@app.route('/search')
def endgame_search():
    return render_template('search.html', title='Search', company_names=company_names)

# The good stuff
@app.route('/output')
def endgame_output():
    """Make the output."""
    company_name = request.args.get('company_name')
    cid = get_company_id(company_name)

    js_resources = None
    css_resources = None
    comps = None
    script = None
    div = None

    # Get prediction class
    is_startup = (cid in pred.index)

    if is_startup:
        pred_class = pred.loc[cid, 'pred']
        if pred_class == 0:
            message = ' will exit in a timely fashion!'
        elif pred_class == 1:
            message = ' will exit, but in a long time.'
        elif pred_class == 2:
            message = ' will continue operating without exit.'
        else:
            message = ': Oops! It doesn\'t look like we have a prediction for that company.'

        # Get list of similar company ids
        try:
            si = sim_idx[cid] # actual pandas integer indices
            comps = df.loc[si[1:app.config['N_SIMILAR']], ['name', 'category_code', 'dates', 'status']]
        except KeyError as e:
            return render_template('500.html', error=e)

        # Make plot!
        plot = plot_timelines(si[::-1])
        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()
        # Embed plot into HTML via Flask Render
        script, div = components(plot)

    else: # get current company status
        status = df.loc[df.id == cid].status.values
        label = y.loc[y.id == cid].label.values
        if status == 'operating':
            message = ' continues to operate.'
        elif status == 'ipo':
            the_date = tf.loc[(tf.id == cid) & (tf.event_id == 'public')]\
                         .dates.dt.strftime('%m/%d/%Y')
            message = ' went public on {}!'.format(the_date.values[0])
            if label == 0:
                message += ' ' + company_name + ' exited before the median age for its industry.'
            elif label == 1:
                message += ' ' + company_name + ' exited after the median age for its industry.'
        elif status == 'acquired':
            the_date = tf.loc[(tf.id == cid) & (tf.event_id == 'acquired')]\
                         .dates.dt.strftime('%m/%d/%Y')
            message = ' was acquired on {}!'.format(the_date.values[0])
            if label == 0:
                message += ' ' + company_name + ' exited before the median age for its industry.'
            elif label == 1:
                message += ' ' + company_name + ' exited after the median age for its industry.'
        else:
            message = ': Oops! It doesn\'t look like we have any information on that company.'

    # Jinja that shit
    company_message = company_name + message


    # TODO fp needs better names, descriptions (footnotes?)
    return render_template('output.html', title='Results', 
                           company_names=company_names,
                           company_message=company_message,
                           is_startup=is_startup,
                           comps=comps, plot_script=script,
                           plot_div=div, js_resources=js_resources,
                           css_resources=css_resources)

# Embed slides in app
@app.route('/about')
def endgame_about():
    return render_template('about.html')

# Place comments here, otherwise jinja still tries to fill them!!
# <!-- <p>The 5 most important features in the analysis were:</p> -->
# <!-- <table class="table table&#45;hover"> -->
# <!--   <tr> <th>Feature</th> <th>Importance</th> </tr> -->
# <!--   {% for feat in fp[:5] %} -->
# <!--   <tr><td>{{ feat[0].replace('_', ' ') }}</td> -->
# <!--     <td>{{ "{:0.2f}%".format(100*feat[1]) }}</td> -->
# <!--   </tr> -->
# <!--   {% endfor %} -->
# <!-- </table> -->
#==============================================================================
#==============================================================================
