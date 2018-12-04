from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.model_selection import KFold

# Iteration 8
def test_4(best_params, X_train, y_train, X_test):
	model = xgb.XGBClassifier(**best_params)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	return y_pred

# Iteration 7
def test_3(best_params, X_train, y_train, X_test):
	model = xgb.XGBClassifier(**best_params)
	kf = KFold(n_splits=5, shuffle=True, random_state=0)
	kf.get_n_splits(X_train)
	train_index, test_index = list(kf.split(X_train))[0]
	model.fit(X_train[train_index], y_train[train_index], early_stopping_rounds=10, eval_metric='error', eval_set=[(X_train[test_index], y_train[test_index])])
	y_pred = model.predict(X_test)
	return y_pred

# Iteration 6
def test_2(X_train, y_train, X_test):
	model = SGDClassifier(loss='log', alpha=0.001, penalty='elasticnet', l1_ratio=1, max_iter=5, tol=None, random_state=0)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	return y_pred

# Iteration 5
def test_1(best_params, X_train, y_train, X_test):
	model = SGDClassifier(**best_params)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	return y_pred

# Iteration 1-4
def test_0(X_train, y_train, X_test):
	model = SGDClassifier(loss='log', max_iter=5, tol=None, random_state=0)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	return y_pred
