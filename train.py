import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
import ml.cv as cv
import ml.greedy as greedy
import json
from sklearn.preprocessing import StandardScaler

LABELS = "../data/accidents_after_inspection.txt"
METRICS = "../data/violation_metrics.csv"
INS_METRICS = "../data/inspection_metrics.csv"
READ_OPTS = {"dtype": {"EVENT_NO": object, "MINE_ID": np.float64}}
INS_DATA = "../data/inspection_with_employee_info.txt"
def main():

	keys = ["MINE_ID", "EVENT_NO"]
	tf = pd.read_csv(INS_METRICS, **READ_OPTS)
	xf = pd.read_csv(METRICS, **READ_OPTS)
	lf = pd.read_csv(LABELS, sep="|", **READ_OPTS)
	df = pd.merge(tf, lf, how="left", on=keys, sort=False)
	df = pd.merge(df, xf, how="left", on=keys, sort=False)

	ignore = ["ACCIDENT_DT", "INSPECTION_BEGIN_DT"] + keys
	usecols = [c for c in df.columns if c not in ignore]
	for c in usecols:
		if df[c].isnull().sum() > 0:
			mval = df[c].mean()
			df[c] = df[c].fillna(mval)

	usecols = [usecols[i] for i in [2, 3, 22, 46, 1, 20, 39]]
	savecols = usecols[:2]
	#print usecols
	df = df.ix[:, usecols + keys + ["ACCIDENT_DT", "INSPECTION_BEGIN_DT"]]

	tf = pd.read_csv(INS_DATA, usecols=range(46,76)+[1,2], sep="|", **READ_OPTS)

	newcols = [c for c in tf.columns if c not in keys]
	df = pd.merge(df, tf, how="left", sort=False, on=keys)
	
	y = 1*(df.ACCIDENT_DT.notnull())
	ac = usecols + newcols
	#ac = [ac[i] for i in [20, 38, 51, 52, 65, 57, 63]]
	#print ac
	#ac = [ac[i] for i in [5, 9, 10, 11, 24, 16, 22]] 
	ac = [ac[i] for i in [5, 9, 10, 11, 24, 16, 22]] + savecols

	df.ix[df.ACCIDENT_DT.isnull(), ac].mean().to_json("means.json")
	df = df.fillna(0)
	bf = df.sort(["MINE_ID", "INSPECTION_BEGIN_DT"])
	bf = df.groupby("MINE_ID").last().reset_index()
	bf.ix[:, ac + keys + ["ACCIDENT_DT", "INSPECTION_BEGIN_DT"]].to_csv("last_data.csv", index=False)

	df = df.fillna(0)
	X = np.array(df.ix[:, ac])
	print y.sum()

	clf = LogisticRegression()
	#clf = LDA()
	#print greedy.MultiGreedyAUC(X, y, clf, usecols)
	#return
	print cv.holdout(X, y, clf, nFolds=5, verbose=True, seed=100)
	clf.fit(X, y)
	int_ = [("Intercept", "%f"%clf.intercept_[0])]
	out_json = dict(int_ + [ (ac[i],"%f"%clf.coef_[0][i]) for i in xrange(len(ac))])
	open('coef.json', 'w').write(json.dumps(out_json))

if __name__ == "__main__":
	main()