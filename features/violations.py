import pandas as pd
import numpy as np
import common as cmn

INS_DT = "INSPECTION_BEGIN_DT"

def count_categories(df, keys, cats=[], threshold=25):
	"""
	"""
	if threshold > 0:
		for f in cats:
			df[f] = cmn.enumerate_strings(df, f, threshold=threshold)

	cols = list(keys)
	for f in cats:
		x = pd.get_dummies(df[f], prefix=f)
		for c in x.columns:
			df[c] = x[c]
			cols.append(c)
		df = df.drop(f,1)

	df["NUM_VIOLATE"] = 1
	cols.append("NUM_VIOLATE")
	xf = df.groupby(keys+[INS_DT]).sum().reset_index()

	xf = xf.sort(["MINE_ID", INS_DT])
	for c in cols:
		if c not in keys:
			xf[c] = xf.groupby("MINE_ID")[c].cumsum()
	return xf

def numeric_data(df, keys, cols=[]):
	"""
	"""
	fns = [np.min, np.max, np.mean]
	fn_list = dict([(c, fns) for c in cols])
	xf = df.groupby(keys).agg(fn_list).reset_index()
	xf.columns = ['_'.join(x) if len(x[1]) > 1 else x[0] for x in xf.columns]
	for c in xf.columns:
		if c in cols:
			xf[c] = cmn.impute(xf, c)
	return xf

def get_violation_data(df, threshold=0):
	"""
	"""
	keys = ["MINE_ID", "EVENT_NO"]
	cat_feats = ["LIKELIHOOD", "INJ_ILLNESS", "NEGLIGENCE", "COAL_METAL_IND"]
	vf = df.ix[:, keys + [INS_DT]+cat_feats].copy()
	vf = count_categories(vf, keys, cats=cat_feats, threshold=threshold)

	num_feats = ["VIOLATOR_INSPECTION_DAY_CNT", "NO_AFFECTED"]
	nf = numeric_data(df.ix[:, keys + num_feats], keys, cols=num_feats)

	return pd.merge(nf, vf, on=keys, how="inner", sort=False)




