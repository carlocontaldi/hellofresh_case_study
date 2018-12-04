import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('whitegrid')
import sklearn.metrics as metrics
import itertools

def amount_density(errors):
	"""Compensation amount densities in terms of refund types."""
	plt.figure()
	plt.title('Compensation Amount Densities in terms of Refund Types')
	errors[errors['compensation_type'].isin(['full_refund', 'partial_refund', 'refund'])].groupby('compensation_type')['compensation_amount'].plot(kind='kde', legend=True, xticks=range(0, 151, 10), xlim=(0, 151))
	plt.show()

def categorical_distros(boxes, cancels, errors):
	"""Categorical value distributions."""
	fig = plt.figure()
	fig.suptitle('Categorical Value Distributions')
	plt.subplot(2, 2, 1)
	products = boxes['product'].value_counts().sort_index()
	sns.barplot(y=np.arange(1, len(products)+1), x=products, orient='h')
	plt.title('Boxes - Product Types')
	plt.xlabel('')
	channels = boxes.loc[boxes['subscription_id'].drop_duplicates().index]['channel'].value_counts()
	channels = channels.reindex(index=channels.index.to_series().str[7:].astype(int).sort_values().index)
	plt.subplot(2, 2, 2)
	sns.barplot(y=np.arange(1, len(channels)+1), x=channels, orient='h')
	plt.yticks(range(0, len(channels), 3), range(1, len(channels)+1, 3))
	plt.title('Boxes - Channels')
	plt.xlabel('')
	plt.subplot(2, 2, 3)
	event_types = cancels['event_type'].value_counts()
	sns.barplot(x=event_types.index, y=event_types)
	plt.title('Cancels - Event Types')
	plt.ylabel('')
	plt.subplot(2, 2, 4)
	comp_types = errors['compensation_type'].value_counts()
	sns.barplot(y=comp_types.index, x=comp_types, orient='h')
	plt.title('Errors - Compensation Types')
	plt.xlabel('')
	fig.subplots_adjust(hspace=0.4)
	fig.subplots_adjust(wspace=0.6)
	plt.show()

def event_density(entries):
	"""Event density over time."""
	fig = sns.kdeplot(entries.index.values, bw=0.5)
	x = np.linspace(entries.index.min().value, entries.index.max().value, 11)
	x = [pd.Timestamp(e) for e in x]
	x = ['-'.join([str(e.year), str(e.month)]) for e in x]
	fig.set_xticklabels(x)
	fig.set_title('Event Density over Time')
	plt.show()

def mode_ratios(ratios):
	"""Box type mode ratios."""
	ratios.plot(kind='kde')
	plt.title('Boxes Type Mode Ratios')
	plt.show()

def churn_hist(df, deltags, s):
	"""Churn stacked histogram."""
	fig = df['churned'].groupby(df.index.get_level_values(0)).value_counts().unstack().plot.bar(stacked=True)
	fig.set_xticklabels(['-'.join([str(e.year), str(e.month)]) for e in deltags])
	fig.set_title(s.join(['Churn Rate Over Time - ', ' Truncation']))
	plt.show()

def confusion_matrix(y_test,  y_pred,  title):
	"""Confusion matrix."""
	cm = metrics.confusion_matrix(y_test, y_pred)
	plt.matshow(cm, cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('Churned')
	plt.xlabel('Predicted')
	plt.title(title)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	plt.show()
