import numpy as np
import pandas as pd
import datetime

def extract_event(df, events, gran, deltag_index, date, output):
	"""Helper function useful to identify the occurrence of events within specified time ranges."""
	tmp_events = events.copy()
	tmp_events['range_start_idx'] = np.piecewise(np.zeros(len(tmp_events)),  [(tmp_events[date].values>=start)&(tmp_events[date].values<end) for start, end in
		zip(deltag_index['range_start'].values, deltag_index['range_start'].values+np.timedelta64(gran))], deltag_index.index.values)
	tmp_events = tmp_events[tmp_events['range_start_idx']>0]
	tmp_events = pd.merge(tmp_events, deltag_index, left_on='range_start_idx', right_index=True)
	tmp_events = tmp_events.drop(columns=['range_start_idx'])
	tmp_events[output] = 1
	tmp_events = tmp_events.drop_duplicates(subset=['range_start', 'subscription_id'])
	# df = df.merge(tmp_events, left_on=['delivery_date', 'subscription_id'], right_on=['range_start', 'subscription_id'], how='left').set_index(df.index)
	df = df.merge(tmp_events, left_index=True, right_on=['range_start', 'subscription_id'], how='left').set_index(df.index)
	df = df.fillna(0)
	return df

def extract_pauses(df, pauses, deltag_index, gran):
	"""extract_pauses_v2 - Defines a new column telling whether each customer paused his subscription during the specified time period. In particular, if a pause period of a customer intersects the timedelta of a related customer entry, then the 'paused' field of the entry is set."""
	df = extract_event(df, pauses, gran, deltag_index, 'pause_start', 'paused')
	df = df.drop(columns=['pause_start', 'pause_end', 'range_start', 'subscription_id'])
	df = extract_event(df, pauses, gran, deltag_index, 'pause_end', 'paused2')
	df = df.drop(columns=['pause_start', 'pause_end', 'range_start', 'subscription_id'])
	df['paused'] = df[['paused', 'paused2']].max(axis=1)
	df = df.drop(columns=['paused2'])
	return df

def extract_cancels(df, cancels, deltag_index, gran):
	"""Defines two new columns telling whether each customer cancelled or reactivated his subscription during the specified time period. In particular, if a cancellation event of a customer occurs within the timedelta of a related customer entry, then the 'cancelled'/'reactivated' field of the entry is set."""
	df = extract_event(df, cancels[cancels['event_type']=='cancellation'], gran, deltag_index, 'event_date','cancelled')
	df = df.drop(columns=['event_type', 'event_date','range_start', 'subscription_id'])
	df = extract_event(df, cancels[cancels['event_type']=='reactivation'], gran, deltag_index, 'event_date', 'reactivated')
	df = df.drop(columns=['event_type', 'event_date', 'range_start', 'subscription_id'])
	return df

def extract_errors(df, errors, deltag_index, gran):
	"""Define a one-hot column per each compensation_type telling whether each customer experienced an error during the specified time period and how HelloFresh compensated him for the issue. This function also defines an additional column reporting the total compensation_amount received by the customer in the specified time period."""
	df['amount'] = 0
	for c_type in errors['compensation_type'].unique():
		df = extract_event(df, errors[errors['compensation_type']==c_type], gran, deltag_index, 'week', c_type)
		df['amount'] = df['amount'] + df['compensation_amount']
		df = df.drop(columns=['reported_date', 'week', 'compensation_type', 'compensation_amount', 'range_start', 'subscription_id'])
	return df

def extract_from_subscriptions(df, gb_boxes):
	"""Define a column reporting the time period between customer subscription start and the beginning of the specified time period, in days."""
	from_subscription = gb_boxes['delivery_date'].first() - gb_boxes['started_week'].first()
	from_subscription.name = 'from_subscription'
	df = df.join(from_subscription)
	df['tmp'] = datetime.timedelta(0)
	df['from_subscription'] = df[['from_subscription', 'tmp']].max(axis=1)	# replace negative timedeltas with 0s
	df = df.drop(columns=['tmp'])
	return df

def extract_channels(df, gb_boxes):
	"""Define a column reporting the marketing channel through which the customer was acquired."""
	channels = gb_boxes['channel'].first().str[7:]
	channels.name = 'channel'
	df = df.join(channels)
	return df

def extract_box_types(df, gb_boxes):
	"""Define a column reporting the mode of the boxes delivered to the customer in the specified time period, in terms of box type."""
	prods_by_sub = gb_boxes['product']
	mode_ratios = (prods_by_sub.value_counts()/prods_by_sub.count()).unstack().max(axis=1)
	fav_products = gb_boxes['product'].value_counts().unstack().idxmax(axis=1).str[4:]
	fav_products.name = 'box_type'
	df = df.join(fav_products)
	return df, mode_ratios

def extract_churns(df, deltags, rel_delta_churn):
	"""Define a boolean column reporting whether the customer will churn from the beginning of the specified time period by the specified churn period."""
	df['churned'] = 1.0
	churn_ref = df[df['n_boxes']>0]['n_boxes']
	d_len = len(deltags)
	for deltag_i, last_deltag_i in zip(range(d_len-1-rel_delta_churn, -1, -1), range(d_len-1, rel_delta_churn-1, -1)):
		deltag = deltags[deltag_i]
		for next_deltag in deltags[deltag_i+1:last_deltag_i+1]:
			try:
				intersection = np.intersect1d(df.loc[deltag].index.values, churn_ref.loc[next_deltag].index.values)
				df['churned'].loc[deltag].loc[intersection] = 0
			except KeyError:
				continue
	return df

## First implementations of extract_events
# def extract_pauses_v1(df, pauses, gran):
	# deltag_index = pd.DataFrame({'range_start':df.index.get_level_values(0).drop_duplicates()})
	# df['paused'] = 0
	# diLen = len(deltag_index)
	# pauses = pauses.sort_values('pause_start')
	# i=di=0
	# while pauses.iloc[i][1]<deltag_index.iloc[di]:
		# i+=1
	# while i<len(pauses) and di<diLen:
		# if pauses.iloc[i][1]>=deltag_index.iloc[di]+gran:
			# di+=1
			# continue
		# if pauses.iloc[i][0] in df.loc[deltag_index.iloc[di]].index:
			# df['paused'].loc[deltag_index.iloc[di], pauses.iloc[i][0]] = 1
		# if di<diLen-1 and pauses.iloc[i][2]>=deltag_index.iloc[di+1]:
			# if pauses.iloc[i][0] in df.loc[deltag_index.iloc[di+1]].index:
				# df['paused'].loc[deltag_index.iloc[di+1], pauses.iloc[i][0]] = 1
		# i+=1
		# if i%10000==0:
			# print(i)
	# return df

# def get_inds(deltag_index, date):
	# inds = set()
	# di = deltag_index.searchsorted(date)[0]
	# if di<len(deltag_index):
		# if deltag_index.iloc[di]==date:
			# inds.add(di)
		# elif di>0 and deltag_index.iloc[di]>date:
			# inds.add(di-1)
	# return inds

# def extract_pauses_v0(df, pauses):
	# deltag_index = pd.Series(df.index.get_level_values(0)).drop_duplicates()
	# df['paused'] = 0
	# i=0
	# for e in pauses.iterrows():
		# i+=1
		# if i%100==0:
			# print(i)
		# inds = get_inds(deltag_index, e[1][1])
		# inds.update(get_inds(deltag_index, e[1][2]))
		# for i in inds:
			# if e[1][0] in df.loc[deltag_index.iloc[i]].index:
				# df['paused'].loc[deltag_index.iloc[i], e[1][0]] = 1
	# return df
