import numpy as np
import pandas as pd
import datetime
from code import extract_helper, visualize

def extract(data, churn=datetime.timedelta(weeks=4), gran=datetime.timedelta(weeks=4)):
	"""Feature-engineer a customer dataset by aggregating event data and generating churn labels"""
	print('\nExtract - Extract Event Entries')
	boxes, pauses, cancels, errors = data
	ind_boxes, ind_pauses, ind_pauses2, ind_cancels, ind_errors = boxes.copy(), pauses.copy(), pauses.copy(), cancels.copy(), errors.copy()
	ind_boxes.index = boxes['delivery_date']
	ind_pauses.index = pauses['pause_start']
	ind_pauses2.index = pauses['pause_end']
	ind_cancels.index = cancels['event_date']
	ind_errors.index = errors['week']
	entries = pd.concat([ind_boxes, ind_pauses, ind_pauses2, ind_cancels, ind_errors], sort=False)
	visualize.event_density(entries)
	percentiles = pd.qcut(entries.index, 100)
	print('Percentile ranges:', percentiles.categories, sep='\n')
	entries = entries.loc[entries.index>='2014-10-20 00:00:00']
	entries = entries.loc[entries.index<='2017-12-01 00:00:00']
	print('Events outside of the range [2014-10-20, 2017-12-01] have been removed from the dataset.')

	print('\nExtract - Group By Granularity & Customer')
	gb_entries = entries.groupby([pd.Grouper(freq=gran),'subscription_id'])
	n_boxes = gb_entries['box_id'].count()
	df = pd.DataFrame({'n_boxes':n_boxes}, index=n_boxes.index)
	df.name = 'customers'

	print('\nExtract - Extract & Aggregate Event Information')
	deltags = df.index.get_level_values(0).drop_duplicates()
	deltag_index = pd.DataFrame({'range_start':deltags}, index = range(1, len(deltags)+1))
	df = extract_helper.extract_pauses(df, pauses, deltag_index, gran)
	df = extract_helper.extract_cancels(df, cancels, deltag_index, gran)
	df = extract_helper.extract_errors(df, errors, deltag_index, gran)
	gb_boxes = boxes.groupby('subscription_id')
	df = extract_helper.extract_from_subscriptions(df, gb_boxes)
	df = extract_helper.extract_channels(df, gb_boxes)
	df, mode_ratios = extract_helper.extract_box_types(df, gb_boxes)
	visualize.mode_ratios(mode_ratios)
	rel_delta_churn = round(churn/gran)		# churn period in terms of deltags
	df = extract_helper.extract_churns(df, deltags, rel_delta_churn)
	visualize.churn_hist(df, deltags, 'Before')
	df = df.loc[df.index.get_level_values(0)<deltags[-rel_delta_churn]]
	visualize.churn_hist(df, deltags, 'After')
	print('Average Churn Rate: ', round(df[df['churned']==1]['churned'].count()/len(df), 2))

	return df
