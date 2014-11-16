import pandas as pd
import numpy as np

def enumerate_strings(df, key, threshold=25):
	""""""
	xf = df[key].fillna("unk")
	unq = xf.value_counts()
	index = np.arange(unq.size, dtype='int32')
	cutoff = np.max(index[unq.values >= threshold])
	index[unq.values < threshold] = cutoff + 1
	xf = pd.merge(pd.DataFrame(xf), 
		pd.DataFrame({key: unq.index, key + "Ids": index}),
		on=key,
		how="inner",
		sort=False)

	return xf[key + "Ids"]

def impute(df, key, func=np.mean):
	""""""
	val = func(df[key][df[key].notnull()])
	return df[key].fillna(val)	
