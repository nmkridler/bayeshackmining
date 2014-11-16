import cv
from multiprocessing import Pool

def feature_score_auc(x):
	""""""
	X, y, clf = x
	return cv.holdout(X, y, clf, verbose=False)

def MultiGreedyAUC(X, y, clf, fnames):
	"""
		Do a greedy search maximizing AUC
		Args:
			X: features
			y: labels
			clf: sklearn classifier
			fnames: column names
	"""
	pool = Pool(processes=4)

	bestFeats = []
	lastScore = 0
	allFeats = [f for f in xrange(X.shape[1])]

	while True:
		testFeatSets = [[f] + bestFeats for f in allFeats if f not in bestFeats]
		args = [(X[:,fSet], y, clf) for fSet in testFeatSets]
		scores = pool.map(feature_score_auc, args)
		(score, featureSet) = max(zip(scores,testFeatSets))
		print featureSet
		print "Max AUC: %f"%score
		if score < lastScore:
			break
		lastScore = score
		bestFeats = featureSet

	pool.close()
	return [fnames[i] for i in bestFeats]