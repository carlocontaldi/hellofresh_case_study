from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from scipy.sparse import issparse

# Iteration 6
def preprocess_2(X, y, train, scaler=None, normalizer=None, encoder=None, pca=None):
	y = y.astype(float)
	X[:, 0] = X[:, 0].astype(float)
	X[:, 9] = [float(e.days) for e in X[:, 9]]
	if train:
		print('Preprocess - RobustScaler, MinMaxScaler, OneHotEncoder, PCA')
		scaler = RobustScaler(with_centering=False)
		X[:, [0,4,9]] = scaler.fit_transform(X[:, [0,4,9]])
		normalizer = MinMaxScaler()
		X[:, [0,4,9]] = normalizer.fit_transform(X[:, [0,4,9]].astype(float))
		encoder = OneHotEncoder(categorical_features=[10,11])
		X = encoder.fit_transform(X)
		X = X.toarray() if issparse(X) else X.astype(float)
		pca = PCA(0.99, random_state=0)
		X = pca.fit_transform(X)
		return (X, y), scaler, normalizer, encoder, pca
	else:
		X[:, [0,4,9]] = scaler.transform(X[:, [0,4,9]])
		X[:, [0,4,9]] = normalizer.transform(X[:, [0,4,9]])
		X = encoder.transform(X)
		X = X.toarray() if issparse(X) else X.astype(float)
		X = pca.transform(X)
		return (X, y)

# Iteration 4-5
def preprocess_1(X, y, train, scaler=None, normalizer=None, encoder=None):
	y = y.astype(float)
	X[:, 0] = X[:, 0].astype(float)
	X[:, 9] = [float(e.days) for e in X[:, 9]]
	if train:
		print('Preprocess - RobustScaler, MinMaxScaler, OneHotEncoder')
		scaler = RobustScaler(with_centering=False)
		X[:, [0,4,9]] = scaler.fit_transform(X[:, [0,4,9]])
		normalizer = MinMaxScaler()
		X[:, [0,4,9]] = normalizer.fit_transform(X[:, [0,4,9]].astype(float))
		encoder = OneHotEncoder(categorical_features=[10,11])
		X = encoder.fit_transform(X)
		X = X.toarray() if issparse(X) else X.astype(float)
		return (X, y), scaler, normalizer, encoder
	else:
		X[:, [0,4,9]] = scaler.transform(X[:, [0,4,9]])
		X[:, [0,4,9]] = normalizer.transform(X[:, [0,4,9]])
		X = encoder.transform(X)
		X = X.toarray() if issparse(X) else X.astype(float)
		return (X, y)

# Iteration 1-3
def preprocess_0(X, y, train, scaler=None, normalizer=None, encoder=None):
	y = y.astype(float)
	X[:, 0] = X[:, 0].astype(float)
	X[:, 9] = [float(e.days) for e in X[:, 9]]
	if train:
		print('Preprocess - RobustScaler, MinMaxScaler, OneHotEncoder')
		scaler = RobustScaler()
		X[:, [0,4,9]] = scaler.fit_transform(X[:, [0,4,9]])
		normalizer = MinMaxScaler()
		X[:, [0,4,9]] = normalizer.fit_transform(X[:, [0,4,9]].astype(float))
		encoder = OneHotEncoder(categorical_features=[10,11])
		X = encoder.fit_transform(X)
		X = X.toarray() if scipy.sparse.issparse(X) else X.astype(float)
		return (X, y), scaler, normalizer, encoder
	else:
		X[:, [0,4,9]] = scaler.transform(X[:, [0,4,9]])
		X[:, [0,4,9]] = normalizer.transform(X[:, [0,4,9]])
		X = encoder.transform(X)
		X = X.toarray() if scipy.sparse.issparse(X) else X.astype(float)
		return (X, y)
