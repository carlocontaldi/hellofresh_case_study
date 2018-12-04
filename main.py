import sys
import datetime
from code import *
import pickle

def main(test_size=0.01, churn_period=datetime.timedelta(weeks=4), granularity=datetime.timedelta(weeks=4)):
	data = import_data.import_data()
	data = eda.eda(data)
	df = extract.extract(data, churn_period, granularity)
	with open('df.pkl', 'wb') as f:
		pickle.dump(df, f)		# save feature-engineered dataset
	with open('df.pkl', 'rb') as f:
		df = pickle.load(f)		# load feature-engineered dataset
	(X_train, y_train), (X_test, y_test) = split.split(df, test_size)
	(X_train, y_train), encoder, scaler, normalizer = preprocess.preprocess_1(X_train, y_train, True)
	(X_test, y_test) = preprocess.preprocess_1(X_test, y_test, False, encoder, scaler, normalizer)
	y_pred = evaluate.eval_4(X_train, y_train, X_test, y_test)
	with open('y_pred.pkl', 'wb') as f:
		pickle.dump(y_pred, f)		# useful to evaluate statistical significance
	return y_pred

if __name__ == "__main__":

	class Logger(object):
		def __init__(self):
			self.terminal = sys.stdout
			self.log = open("logfile.log", "w")

		def write(self, message):
			self.terminal.write(message)
			self.log.write(message)

		def flush(self):
			#this flush method is needed for python 3 compatibility.
			#this handles the flush command by doing nothing.
			pass

	sys.stdout = Logger()
	start_time = datetime.datetime.now()
	print('Main execution started at:', start_time)
	main(0.1)
	end_time = datetime.datetime.now()
	print('Main execution ended at: {}.\nTotal execution duration: {}.'.format(end_time, end_time-start_time))
