from sklearn.metrics import mean_absolute_error, roc_curve, auc, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

def auc_score(t, y):
	"""
	"""
	fpr, tpr, th = roc_curve(t, y)
	return auc(fpr, tpr)

def cross_validate(X, y, clf, folds):
	"""
	"""
	yp = np.empty(y.size)
	for train, test in folds:
		clf.fit(X[train, :], y[train])
		y_ = clf.predict_proba(X[test,:])[:,1]

	return yp

def k_fold(X, y, clf, nFolds=5, seed=1337):
	"""
		Perform stratified k-fold cross validation
		Args:
			X: features
			y: labels
			clf: sklearn classifier object
			nFolds: number of folds
			seed: random seed
		Returns:
			probabilities
	"""
	np.random.seed(seed)
	kf = KFold(len(y), n_folds=nFolds, indices=True, shuffle=True, random_state=1337)
	return cross_validate(X, y, clf, kf)


def stratified_k_fold(X, y, clf, nFolds=5, seed=1337):
	"""
		Perform stratified k-fold cross validation
		Args:
			X: features
			y: labels
			clf: sklearn classifier object
			nFolds: number of folds
			seed: random seed
		Returns:
			probabilities
	"""
	np.random.seed(seed)
	kf = StratifiedKFold(y, n_folds=nFolds)
	return cross_validate(X, y, clf, kf)

def holdout(X, y, clf, fraction=0.2, nFolds=10, seed=1337, verbose=True):
	""""""
	meanScore = 0.
	for i in xrange(nFolds):
		xTr, xCV, yTr, yCV = train_test_split(X,y,
			test_size=fraction,random_state=i*seed)
		clf.fit(xTr,yTr)
		y_ = clf.predict_proba(xCV)[:,1]
		thisScore = auc_score(yCV, y_)
		meanScore += thisScore
		if verbose:
			print "Error: %f (fold %d of %d)"%(thisScore, i, nFolds)

	meanScore /= nFolds
	return meanScore

