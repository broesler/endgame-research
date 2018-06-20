#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: phreg_statsmodels_ex.py
#  Created: 06/19/2018, 12:17
#   Author: Bernie Roesler
#
"""
  Description:
"""
#==============================================================================
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = sm.datasets.get_rdataset("flchain", "survival").data
del data["chapter"]
data = data.dropna()
data["lam"] = data["lambda"]
data["female"] = (data["sex"] == "F").astype(int)
data["year"] = data["sample.yr"] - min(data["sample.yr"])
status = data["death"].values

mod = smf.phreg("futime ~ 0 + age + female + creatinine + "
                "np.sqrt(kappa) + np.sqrt(lam) + year + mgus",
                data, status=status, ties="efron")
rslt = mod.fit()
print(rslt.summary())

#==============================================================================
#==============================================================================
