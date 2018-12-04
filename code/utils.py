import datetime

def hfWeek2datetime(s):
	"""Convert an HelloFresh week into the datetime associated with its start."""
	if s[3]>='6':
		return datetime.datetime.strptime(s + '-6', "%Y-W%W-%w") - datetime.timedelta(weeks=1)
	else:
		return datetime.datetime.strptime(s + '-6', "%Y-W%W-%w") - datetime.timedelta(weeks=2)

def replaceRefund(e):
	"""Redistribute compensation type values."""
	e['compensation_type'] = 'partial_refund' if e['compensation_amount']<55.0 else 'full_refund'
	return e
