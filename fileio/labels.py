import pandas as pd
import numpy as np

READ_OPTS = {"sep": "|"}

def inspections(filename):
	"""
		EVENT_NO
		MINE_ID

		INSPECTION_BEGIN_DT
		INSPECTION_END_DT
	"""
	opts = READ_OPTS.copy()
	opts["parse_dates"] = [ "INSPECTION_BEGIN_DT",
							"INSPECTION_END_DT"]
	opts["usecols"] = opts["parse_dates"] + ["EVENT_NO", "MINE_ID", "CAL_YR"]
	return pd.read_table(filename, **opts)

def accidents_by_dates(filename):
	"""
		Return a list of accidents
			MINE_ID
			ACCIDENT_TIME
			ACCIDENT_DT
	"""
	opts = READ_OPTS.copy()
	opts["parse_dates"] = ["ACCIDENT_DT"]
	opts["usecols"] = opts["parse_dates"] + ["MINE_ID", "ACCIDENT_TIME", "CAL_YR"]
	return pd.read_table(filename, **opts)

def get_accidents(accident_file, inspection_file):
	"""
	"""
	acc = accidents_by_dates(accident_file)
	acc["PREV_CAL_YR"] = acc["CAL_YR"] - 1
	ins = inspections(inspection_file)

	acc_this_year = pd.merge(acc, ins, on=["MINE_ID","CAL_YR"], how="left", sort=False)
	new_cols = {"CAL_YR": "PREV_CAL_YR", "PREV_CAL_YR": "CAL_YR"}
	acc_last_year = pd.merge(acc.rename(columns=new_cols), ins, 
		on=["MINE_ID","CAL_YR"], how="left", sort=False)

	df = acc_this_year.append(acc_last_year, ignore_index=True)
	df.INSPECTION_END_DT[df.INSPECTION_END_DT.isnull()] = (
		df.INSPECTION_BEGIN_DT[df.INSPECTION_END_DT.isnull()])
	df = df.loc[df.ACCIDENT_DT > df.INSPECTION_END_DT]
	levels = ["MINE_ID", "ACCIDENT_DT", "ACCIDENT_TIME"]
	return df.groupby(levels).apply(
		lambda x: x.iloc[x.INSPECTION_END_DT.values.argmax()])
