import math
import numpy as np
import pandas as pd
from scipy.stats import beta
import random
import datetime
import math
import time
import bisect
random_seed = 1
np.random.seed(random_seed)

class EP4:
	def __init__(self, x, y):
		#array, array, int -> None
		'''We initially receive the x and y array
		which are expected to have dimension 3 each
		used to calculate the Dirichlet distribution
		'''
		self.x = x
		self.y = y
		#with 95% confidence
		# z = 1.96
		# sigma^2 = 0.25 in worst case scenario where estimate is 0.5 and variance is p(1 -p) = 0.25
		# 0.0005 is the absolute error tolerated
		#n = z^2 * sigma^2 / 0.0005^2
		# self.n = 2722500 #defined in the report
		self.n = 10000000 

		self.alpha = [x[i] + y[i] for i in range(len(x))]

		#denominator_constant is the denominator of the f(Theta) function
		# which is 1/B(x + y) where B(x + y) can be expressed as
		# B(alpha) = ( product gamma(alpha) ) / gamma( sum (alpha) )
		# https://en.wikipedia.org/wiki/Dirichlet_distribution#Probability_density_function
		numerator = 1
		for i in range(len(self.alpha)):
			numerator *= math.gamma(self.alpha[i])
		B_x_y = numerator / math.gamma(sum(self.alpha))

		self.denominator_constant = B_x_y

	def generate_theta(self):
		#None -> None
		thetas = np.random.dirichlet(self.alpha, self.n)
		self.thetas = thetas

	def f(self, theta):
		#array -> float
		'''
		Receives an array of 
		dimension 3 with the Theta values
		generated randomly by the Dirichlet
		distribution and returns
		the value of f (as defined
		in the problem statement)
		for those Theta
		'''
		numerator = 1
		dim_alpha = len(self.alpha)
		for i in range(dim_alpha):
			numerator *= theta[i]**(self.alpha[i] - 1)
		value = numerator/self.denominator_constant

		return value

	def order_f_thetas(self):
		#None -> None
		'''
		By ordering the values of each f(Theta),
		we can determine the start and end of each
		k-bin by simply taking 1/k % of points to 
		each bin and recording the start and end of each.
		That way, we don't have to dinamically
		set the bin size but rather do so 
		in linear time
		'''
		f_thetas = [self.f(theta) for theta in self.thetas]
		f_thetas.sort()

		self.ordered_f_thetas = f_thetas
		self.min_f = f_thetas[0] #min value of f_thetas since it is ordered
		self.sup_f = f_thetas[-1] #max value of f_thetas since it is ordered

	def U(self, v):

		if v > self.sup_f:
			return 1
		if v < self.min_f:
			return 0

		n = self.n
		'''
		The idea behind this method is:
		Since we have an ordered list the the values of f(theta)
		for each theta in our sample space,
		then we need to find out how many
		theta observations have f(theta) below certain v.
		By finding the index where we would insert
		a new observation of value v in our 
		ordered list of f(thetas), we use that index
		to determine how many observations are to the left
		(lower values). The number of observations whose
		f(theta) values are below v divided by the 
		total number of observations (n) is the estimate
		for W(v).
		'''

		#https://docs.python.org/3/library/bisect.html?highlight=insort#bisect.bisect_left
		# The returned insertion point i partitions the array a into two halves 
		# so that all(val < x for val in a[lo:i]) for the left side and 
		# all(val >= x for val in a[i:hi]) for the right side.
		i = bisect.bisect_left(self.ordered_f_thetas, v)

		# i + 1 because i starts at 0
		return (i + 1)/n #index divided by total n points
