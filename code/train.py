from sklearn.model_selection import KFold, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import SGDClassifier
import xgboost as xgb

# Iteration 8
def train_4(X_train, y_train):
	model = xgb.XGBClassifier()
	best_params = dict(learning_rate=0.05, n_estimators=15, random_state=0)
	kf = KFold(n_splits=5, shuffle=True, random_state=0)
	kf.get_n_splits(X_train)
	y_pred = y_train.copy()
	for train_index, test_index in kf.split(X_train):
		X_t, X_v = X_train[train_index], X_train[test_index]
		y_t = y_train[train_index]
		model = xgb.XGBClassifier(**best_params)
		model.fit(X_t, y_t)
		y_pred[test_index] = model.predict(X_v)
	return best_params, y_pred

# Iteration 7
def train_3(X_train, y_train):
	param_grid = {
	'learning_rate' : [0.02, 0.05, 0.08],
	'n_estimators' : [100],
	'random_state' : [0]
	}
	fit_params = {
		'early_stopping_rounds' : 10,
		'eval_metric' : 'error',
		'eval_set' : [[X_train, y_train]]
	}
	model = xgb.XGBClassifier()
	grid = GridSearchCV(model, param_grid, fit_params=fit_params, cv=TimeSeriesSplit(n_splits=5).get_n_splits([X_train, y_train]))
	grid.fit(X_train, y_train)
	best_params = grid.best_params_
	kf = KFold(n_splits=5, shuffle=True, random_state=0)
	kf.get_n_splits(X_train)
	y_pred = y_train.copy()
	for train_index, test_index in kf.split(X_train):
		X_t, X_v = X_train[train_index], X_train[test_index]
		y_t, y_v = y_train[train_index], y_train[test_index]
		model = xgb.XGBClassifier(**best_params)
		model.fit(X_t, y_t, early_stopping_rounds=10, eval_metric='error', eval_set=[(X_v, y_v)])
		y_pred[test_index] = model.predict(X_v)
	return best_params, y_pred

# Iteration 6
def train_2(X_train, y_train):
	kf = KFold(n_splits=10, shuffle=True, random_state=0)
	kf.get_n_splits(X_train)
	y_pred = y_train.copy()
	for train_index, test_index in kf.split(X_train):
		X_t, X_v = X_train[train_index], X_train[test_index]
		y_t = y_train[train_index]
		model = SGDClassifier(loss='log', alpha=0.001, penalty='elasticnet', l1_ratio=1, max_iter=5, tol=None, random_state=0)
		model.fit(X_t, y_t)
		y_pred[test_index] = model.predict(X_v)
	return y_pred

# Iteration 5
def train_1(X_train, y_train):
	param_grid = {
    'loss': ['log'],
    'penalty': ['elasticnet'],
    'alpha': [10 ** x for x in range(-6, 1)],
    'l1_ratio': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1],
	'max_iter': [5],
	'tol': [None],
 	'random_state': [0]
}
	grid = GridSearchCV(SGDClassifier(), param_grid, cv=5, scoring='accuracy')
	grid.fit(X_train, y_train)
	best_params = grid.best_params_
	kf = KFold(n_splits=10, shuffle=True)
	kf.get_n_splits(X_train)
	y_pred = y_train.copy()
	for train_index, test_index in kf.split(X_train):
		X_t, X_v = X_train[train_index], X_train[test_index]
		y_t = y_train[train_index]
		model = SGDClassifier(**best_params)
		model.fit(X_t, y_t)
		y_pred[test_index] = model.predict(X_v)
	return best_params, y_pred

# Iteration 1-4
def train_0(X_train, y_train):
	kf = KFold(n_splits=10, shuffle=True, random_state=0)
	kf.get_n_splits(X_train)
	y_pred = y_train.copy()
	for train_index, test_index in kf.split(X_train):
		X_t, X_v = X_train[train_index], X_train[test_index]
		y_t = y_train[train_index]
		model = SGDClassifier(loss='log', max_iter=5, tol=None, random_state=0)
		model.fit(X_t, y_t)
		y_pred[test_index] = model.predict(X_v)
	return y_pred
