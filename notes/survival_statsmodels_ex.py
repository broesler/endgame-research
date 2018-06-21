#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: survival_statsmodels_ex.py
#  Created: 06/19/2018, 10:50
#   Author: Bernie Roesler
#
"""
  Description: Statsmodels Survival Analysis Example
    See: <http://www.statsmodels.org/0.8.0/duration.html>
"""
#==============================================================================

import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.close('all')

data = sm.datasets.get_rdataset("flchain", "survival").data
df = data.loc[data.sex == "F", :]
sf = sm.SurvfuncRight(df["futime"], df["death"])

print(sf.summary().head())
# print(sf.quantile(0.25))
# print(sf.quantile_ci(0.25))

# Plot single survival curve, remove censoring symbols
fig = sf.plot()
ax = fig.get_axes()[0]
pt = ax.get_lines()[1]
pt.set_visible(False)

# Add 95% confidence interval
lcb, ucb = sf.simultaneous_cb()
ax = fig.get_axes()[0]
ax.fill_between(sf.surv_times, lcb, ucb, color='lightgrey')
# ax.set_xlim(365, 365*10)  # limit CI to center of curves
# ax.set_ylim(0.7, 1)
ax.set_ylabel("Proportion alive")
ax.set_xlabel("Days since enrollment")

# Multiple survivals
plt.figure(2)
gb = data.groupby("sex")
ax = plt.axes()
sexes = []
for g in gb:
    sexes.append(g[0])
    sf = sm.SurvfuncRight(g[1]["futime"], g[1]["death"])
    sf.plot(ax)
li = ax.get_lines()
li[1].set_visible(False)
li[3].set_visible(False)
plt.figlegend((li[0], li[2]), sexes, "center right")
plt.ylim(0.6, 1)
ax.set_ylabel("Proportion alive")
ax.set_xlabel("Days since enrollment")

# Compare two survival curves. The default procedure is the logrank test:
# See kwarg "weight_type" for more options
stat, pv = sm.duration.survdiff(data.futime, data.death, data.sex)

plt.show()
#==============================================================================
#==============================================================================
