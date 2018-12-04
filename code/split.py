import numpy as np

def split(df, test_size=0.01):
	"""Split available data into Training & Development set and hold-out Test set."""
	print('\nSplit - Train&Dev Size = ', 1-test_size, ' , Test Size = ', test_size, sep='')
	df.index = df.index.droplevel(level=1)		# dropping subscription_id
	test_len = round(test_size*len(df))
	df_train, df_test = df.iloc[:len(df)-test_len], df.iloc[-test_len:]
	df_train = df_train.reset_index().drop(columns=['index'])
	df_test = df_test.reset_index().drop(columns=['index'])
	X_train = np.array(df_train.drop(columns=['churned']))
	X_test = np.array(df_test.drop(columns=['churned']))
	y_train = np.array(df_train['churned'])
	y_test = np.array(df_test['churned'])
	print('Churn rates in train and test sets:', round(y_train.sum()/len(y_train), 2), round(y_test.sum()/len(y_test), 2))
	return (X_train, y_train), (X_test, y_test)
