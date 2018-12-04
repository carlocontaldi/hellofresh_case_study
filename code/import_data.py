import pandas as pd

def import_data():
	boxes = pd.read_csv('./data/boxes.csv')
	pauses = pd.read_csv('./data/pauses.csv')
	cancels = pd.read_csv('./data/cancels.csv')
	errors = pd.read_csv('./data/errors.csv')
	print('Boxes dataset:', boxes.head(), 'Pauses dataset:', pauses.head(),
		'Cancels dataset:', cancels.head(), 'Errors dataset:', errors.head(), sep='\n')
	return boxes, pauses, cancels, errors
