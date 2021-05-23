import random

'''
- Estimate pi by the proportion p=(1/n)sum(T(xi)) 
where T(x) = Ind(||x||2 <= 1) tests if xi falls 
inside the unit circle;
- Set n so to obtain an estimate that is 
accurate to 0.05%
- Write a well documented source code in Python,
and a very nice report in LaTeX explaining 
everything you did, including the criteria use
to set n (requires some thinking and choosing)
'''

def falls_inside_unit_circle(x, y):
	#float, float -> boolean
	'''
	given a point with coordinates
	x, y E {0, 1}^2, determine if 
	point falls inside the quarter 
	unit circle centered at (0, 0)
	'''
	return (x**2 + y**2) <= 1

# def find_n(accuracy_threshold):
# 	#float -> int
# 	'''
	# Treating this as an experiment,and pi 
	# as the estimate mean derived from 
	# the samples, we can define a n 
	# sample size using the given accuracy threshold
	# of 0.0005 as Minimal Detectable Difference (MDD)
	# and a 99.9% confidence in our estimate 
	# (Z-score <= 3.62).
	# Since the probability of event
	# p is unknown, but it is known
	# that p E [0, 1], we can use
	# p = 0.5 to achieve maximum 
	# variance of 0.25 and respect
	# the accuracy_threshold even
	# in the worst case.
 #  As our software falls_inside_unit_circle 
 #  is defined using the quarter unit circle,
 #  we will use mdd := mdd/4
# 	'''
# 	z_score = 3.62
# 	p = 0.5
# 	var = p*(1 - p)
# 	mdd := p - estimate # (0.0005/4)
# 	z_score = mdd/math.sqrt(var/n)
# 	z_score*math.sqrt(var)/math.sqrt(n) = mdd
# 	z_score*math.sqrt(var)/mdd = math.sqrt(n)
# 	n = z_score**2 * var / mdd**2
# 	return n

def experiment(n):
	points_inside = 0
	for i in range(n):
		x = random.random()
		y = random.random()
		points_inside += falls_inside_unit_circle(x, y)
	pi = (points_inside/n)*4
	return pi

n = 2096704

print(experiment(n))
