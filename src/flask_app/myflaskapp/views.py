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
pred, sim_idx, feat_imp, _, tf_fund, y = pickle.load(open(data_path + 'flask_db_full.pkl', 'rb'))
# pred, sim_idx, _, tf_fund, y = pickle.load(open(data_path + 'flask_db_full.pkl', 'rb'))
tf, df = pickle.load(open(data_path + 'cb_input_datasets_full.pkl', 'rb'))
# List of company names for autocompletion (limit to startups!!)
company_names = json.dumps(list(df.loc[df.id.isin(pred.index), 'name'].values))

# cats = ['music', 'photo_video', 'analytics', 'messaging', 'search',
#         'education', 'software', 'cleantech', 'network_hosting', 'web',
#         'social', 'automotive', 'offices', 'transportation', 'news',
#         'fashion', 'legal', 'finance', 'enterprise', 'biotech',
#         'real_estate', 'travel', 'health', 'public_relations', 'medical',
#         'semiconductor', 'ecommerce', 'nanotech', 'hospitality',
#         'consulting', 'manufacturing', 'mobile', 'other', 'advertising',
#         'games_video', 'security', 'design', 'hardware', 'sports']

# convert df.category_code to "display" names 
cats = {'games_video':'video games',
        'network_hosting':'network hosting',
        'photo_video':'photo/video',
        'public_relations':'public relations',
        'real_estate':'real estate'}
for k in cats.keys():
    df.loc[df.category_code == k, 'category_code'] = cats[k]

# Define feature columns to use in feature importance
feat_cols = ['acquisition_events', 'mean_acquisition_time', 'latitude',
             'longitude', 'mean_famt', 'mean_funded_time', 'offices',
             'funded_events', 'mean_milestone_time',  'experience',
             'investment_events', 'milestone_events', 'products',
             'famt_cumsum', 'mean_investment_time']

# Define pretty-print conversions
fc_p = {'acquisition_events':'Acquisitions Made',
        'mean_acquisition_time':'Mean Time To Acquistions Made',
        'latitude':'Company Location: Latitude',
        'longitude':'Company Location: Longitude',
        'mean_famt':'Mean Funding Amount',
        'mean_funded_time':'Mean Time Between Funding Rounds',
        'offices':'Number of Offices',
        'funded_events':'Number of Funding Rounds',
        'mean_milestone_time':'Mean Time Between Milestones',
        'experience':'Years of Founder Experience',
        'investment_events':'Investments Made',
        'milestone_events':'Milestones',
        'products':'Products',
        'famt_cumsum':'Cumulative Funding Received',
        'mean_investment_time':'Mean Time Between Investments'}

# Date of last funding event
tf_last_fund = tf_fund.groupby(['id'], as_index=False).max()

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
    for i, color in zip(range(len(si)), colors):
        s = si[i]
        # Get cid from df
        cid = df.loc[s].id
        the_co = tf_fund.loc[tf_fund.id == cid].sort_values('dates')
        if the_co.empty:
            continue
        # Add [0,0] point
        time_to_funding = pd.Series(0, index=[the_co.index[0]])\
                            .append(the_co.time_to_event) / 365
        raised_amount_usd = pd.Series(0, index=[the_co.index[0]])\
                            .append(the_co.famt_cumsum)
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
@app.route('/index')
def endgame_input():
    # Sort list of companies predicted to exit quickly by probability
    exits = pred.loc[pred.pred == 0].copy()
    exits.sort_values('proba0', ascending=False, inplace=True)
    # Get general info
    top_exits = df.loc[df.id.isin(exits.index)].copy()
    top_exits.rename(columns={'dates':'founding_date'}, inplace=True)
    # Get funding info
    top_exits = top_exits.merge(tf_last_fund.loc[tf_last_fund.id.isin(exits.index)], on='id')
    top_exits.rename(columns={'dates':'last_funding_date'}, inplace=True)
    # Remove odd outcomes
    top_exits = top_exits.loc[~(top_exits.founding_date == top_exits.last_funding_date)]
    # Format display strings and replace NaN values with '-'
    top_exits['famt_str'] = top_exits.famt_cumsum.transform(lambda x: '${:,.2f}'.format(x))
    top_exits.loc[top_exits.famt_str == '$nan', 'famt_str'] = '-'
    # Paginate?!
    # page_start = request.args.get('page_start', app.config['I_START'], type=int)
    # N_DISP = app.config['N_DISPLAY']
    # top_exits = top_exits.iloc[N_DISP*page_start:N_DISP*(page_start+1)]
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
    fp = None

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
            # actual pandas integer indices
            si = sim_idx[cid][:app.config['N_SIMILAR']]
            comps = df.loc[si[1:], ['name', 'category_code', 'dates', 'status']]
        except KeyError as e:
            return render_template('500.html', error=e)

        # Make plot!
        plot = plot_timelines(si[::-1])
        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()
        # Embed plot into HTML via Flask Render
        script, div = components(plot)

        # Feature importance
        fp = feat_imp[cid]
        fp = [t for t in fp if t[0] in feat_cols]

        # Convert values to readable names
        fp = [(fc_p[t[0]], t[1]) for t in fp]
        fp = fp[:app.config['N_FEATURES']]

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
                           is_startup=is_startup,
                           company_message=company_message,
                           fp=fp,
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
