import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
import ml.cv as cv
import ml.greedy as greedy

LABELS = "../data/accidents_after_inspection.txt"
METRICS = "../data/violation_metrics.csv"
INS_METRICS = "../data/inspection_metrics.csv"
READ_OPTS = {"dtype": {"EVENT_NO": object, "MINE_ID": np.float64}}

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

	X = np.array(df.ix[:, usecols])
	y = 1*(df.ACCIDENT_DT.notnull())

	clf = LogisticRegression()
	clf = LDA()
	print greedy.MultiGreedyAUC(X, y, clf, usecols)
	return
	print cv.holdout(X, y, clf, nFolds=5, verbose=True)


if __name__ == "__main__":
	main()