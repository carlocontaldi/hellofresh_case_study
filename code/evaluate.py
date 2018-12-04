from sklearn.metrics import accuracy_score, classification_report
from code import train, test, visualize

# Iteration 8
def eval_4(X_train, y_train, X_test, y_test):
	print('Evaluate')
	best_params, y_pred = train.train_4(X_train, y_train)
	accuracy = accuracy_score(y_train, y_pred)
	print('Validation Accuracy:', round(accuracy*100, 2), '%')
	print(classification_report(y_train, y_pred))
	visualize.confusion_matrix(y_train, y_pred, 'Validation Accuracy')
	y_pred = test.test_4(best_params, X_train, y_train, X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print('Test Accuracy:', round(accuracy*100, 2), '%')
	print(classification_report(y_test, y_pred))
	visualize.confusion_matrix(y_test, y_pred, 'Test Accuracy')
	return y_pred

# Iteration 7
def eval_3(X_train, y_train, X_test, y_test):
	print('Evaluate')
	best_params, y_pred = train.train_3(X_train, y_train)
	accuracy = accuracy_score(y_train, y_pred)
	print('Validation Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_train, y_pred, 'Validation Accuracy')
	y_pred = test.test_3(best_params, X_train, y_train, X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print('Test Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_test, y_pred, 'Test Accuracy')
	return y_pred

# Iteration 6
def eval_2(X_train, y_train, X_test, y_test):
	print('Evaluate')
	y_pred = train.train_2(X_train, y_train)
	accuracy = accuracy_score(y_train, y_pred)
	print('Validation Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_train, y_pred, 'Validation Accuracy')
	y_pred = test.test_2(X_train, y_train, X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print('Test Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_test, y_pred, 'Test Accuracy')
	return y_pred

# Iteration 5
def eval_1(X_train, y_train, X_test, y_test):
	print('Evaluate')
	best_params, y_pred = train.train_1(X_train, y_train)
	accuracy = accuracy_score(y_train, y_pred)
	print('Validation Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_train, y_pred, 'Validation Accuracy')
	y_pred = test.test_1(best_params, X_train, y_train, X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print('Test Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_test, y_pred, 'Test Accuracy')
	return y_pred

# Iteration 1-4
def eval_0(X_train, y_train, X_test, y_test):
	print('Evaluate')
	y_pred = train.train_0(X_train, y_train)
	accuracy = accuracy_score(y_train, y_pred)
	print('Validation Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_train, y_pred, 'Validation Accuracy')
	y_pred = test.test_0(X_train, y_train, X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print('Test Accuracy:', round(accuracy*100, 2), '%')
	visualize.confusion_matrix(y_test, y_pred, 'Test Accuracy')
	return y_pred
