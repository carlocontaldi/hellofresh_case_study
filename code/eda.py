import numpy as np
import pandas as pd
import datetime
from code import utils, visualize

def eda(data):
	boxes, pauses, cancels, errors = data

	print('\nEDA - Boxes')
	print('# Missing values:', boxes.isna().sum(), sep='\n')
	print('#Entries:', len(boxes),', # Unique box ids:', boxes['box_id'].nunique())
	print('Entries with null fields:', boxes[boxes['started_week'].isna()], sep='\n')
	print('Entries with problematic subscription:', boxes[boxes['subscription_id']==1654222], sep='\n')
	print('The started_week is immutable for any subscription: ', all(boxes.groupby('subscription_id')['started_week'].nunique()==1))
	started_week_value = boxes[boxes['subscription_id']==1654222].dropna().iloc[0]['started_week']
	boxes['started_week'] = boxes['started_week'].fillna(started_week_value)
	print('Missing values in boxes have been imputed.')
	print('#Entries:', len(boxes), ', # Unique subscription ids:', boxes['subscription_id'].nunique(),
	', Avg #boxes per customer:', round(len(boxes)/boxes['subscription_id'].nunique(), 1))
	boxes['delivery_date'] = boxes['delivery_date'].astype('datetime64[D]')
	# print('Unique values in started_week:',sorted(boxes['started_week'].unique()),sep='\n')
	boxes['started_week'] = boxes['started_week'].apply(utils.hfWeek2datetime)
	boxes_anomalies = boxes[boxes['delivery_date']<boxes['started_week']]
	print('Entries having start of started_week more recent than delivery_date - timedelta summary:',
	(boxes_anomalies['started_week']-boxes_anomalies['delivery_date']).describe(), sep='\n')
	orig_sz = len(boxes)
	boxes = boxes[boxes['delivery_date']-boxes['started_week']>=datetime.timedelta(weeks=-1)]
	print(orig_sz-len(boxes), 'entries having an anomalous timedelta higher than one week have been dropped.')

	print('\nEDA - Pauses')
	pauses['pause_start'] = pauses['pause_start'].astype('datetime64[D]')
	pauses['pause_end'] = pauses['pause_end'].astype('datetime64[D]')
	pause_anomalies = pauses[pauses['pause_end']-pauses['pause_start']<=datetime.timedelta(0)]
	print('Anomalous pauses timedelta summary:', (pause_anomalies['pause_start']-pause_anomalies['pause_end']).describe(), sep='\n')
	pauses = pauses[pauses['pause_end']>pauses['pause_start']]
	print('Anomalous pause entries have been removed.')

	print('\nEDA - Cancels')
	cancels['event_date'] = cancels['event_date'].astype('datetime64[D]')
	print('Proportions of Cancellation and Reactivation Events:', round(cancels['event_type'].value_counts()/len(cancels), 2), sep='\n')

	print('\nEDA - Errors')
	errors['reported_date'] = errors['reported_date'].astype('datetime64[D]')
	errors = errors.rename(columns={'hellofresh_week_where_error_happened':'week'})
	print('Entries having week=="0000-W00":', errors[errors['week']=='0000-W00'], sep = '\n')
	errors = errors[errors['week']!='0000-W00']
	print('The erroneous entry has been dropped.')
	errors['week'] = errors['week'].apply(utils.hfWeek2datetime)
	errors_anomalies = errors[errors['reported_date']<errors['week']]
	print('Entries having start of error week more recent than reported_date - timedelta summary:',
	(errors_anomalies['week']-errors_anomalies['reported_date']).describe(), sep='\n')
	orig_sz = len(errors)
	errors = errors[errors['reported_date']-errors['week']>=datetime.timedelta(weeks=-1)]
	print(orig_sz-len(errors), 'entries having an anomalous timedelta higher than one week have been dropped.')
	errors['compensation_type'] = errors['compensation_type'].apply(lambda s:'full_refund' if s=='full refund' else s)
	print('"full refund" compensation types have been replaced with "full_refund"')
	print('Compensation type statistics:', errors.groupby('compensation_type')['compensation_amount'].describe(), sep='\n')
	visualize.amount_density(errors)
	n_entries = len(errors[errors['compensation_type']=='refund'])
	refund_index = errors[errors['compensation_type']=='refund'].index
	errors.loc[refund_index] = errors.loc[refund_index].apply(utils.replaceRefund, axis=1)
	print('The', n_entries, 'entries having "refund" as compensation_type have been redistributed to "partial_refund" and "full_refund" groups.')
	n_entries = len(errors[errors['compensation_type']=='sorry'])
	errors = errors[errors['compensation_type']!='sorry']
	print('The', n_entries, 'entries having "sorry" as compensation_type have been removed.')

	print('\nEDA - Global dataset')
	visualize.categorical_distros(boxes, cancels, errors)
	print('Dates have been converted into "datetime64"	objects.')
	pauses = pauses[pauses['subscription_id'].isin(boxes['subscription_id'])]
	cancels = cancels[cancels['subscription_id'].isin(boxes['subscription_id'])]
	errors = errors[errors['subscription_id'].isin(boxes['subscription_id'])]
	print('Subscription_ids in pauses are a subset of subscription_ids in boxes:',
	pauses[~pauses['subscription_id'].isin(boxes['subscription_id'])].empty)
	print('Subscription_ids in cancels are a subset of subscription_ids in boxes:',
	cancels[~cancels['subscription_id'].isin(boxes['subscription_id'])].empty)
	print('Subscription_ids in errors are a subset of subscription_ids in boxes:',
	errors[~errors['subscription_id'].isin(boxes['subscription_id'])].empty)

	return boxes, pauses, cancels, errors
