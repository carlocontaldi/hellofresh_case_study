from scipy.stats import mannwhitneyu

def wilcoxon(a, b):
	"""Get p-value describing the statistical significance of the comparison between input series based on the Mann-Whitney (Wilcoxon) test."""
	u, prob = mannwhitneyu(a, b)
	print(prob)
	return prob
