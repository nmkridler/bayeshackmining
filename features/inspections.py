import pandas as pd
import numpy as np
import common as cmn

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
			if x[c].sum() > 0:
				df[c] = x[c]
				cols.append(c)
			if len(x.columns) == 2:
				break
		df = df.drop(f,1)

	return df.groupby(keys).sum().reset_index()

def inspection_metrics(df, threshold=0):
	"""
	"""
	keys = ["MINE_ID", "EVENT_NO"]
	cat_feats = ["ACTIVITY_CODE", "SURF_UG_MINE", "EXPLOSIVE_STORAGE", "MAJOR_CONSTR"]
	df = df.loc[df.CAL_YR > 2011]
	vf = df.ix[:, keys + cat_feats].copy()
	vf = count_categories(vf, keys, cats=cat_feats, threshold=threshold)

	return vf

