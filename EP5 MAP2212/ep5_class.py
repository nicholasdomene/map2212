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

class EP5:
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
		self.n = 2722500 #defined in the report

		self.alpha = [x[i] + y[i] for i in range(len(x))]


		# Covariance matrix to feed Multinormal Distribution for Theta generation
		covariance_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
		a_0 = sum(self.alpha)
		for i in range(len(self.alpha)):
			for j in range(len(self.alpha)):
				if i == j:
					covariance_matrix[i][j] = (a_0 - self.alpha[i])*self.alpha[i]/((a_0**2)*(a_0 + 1))
				else:
					covariance_matrix[i][j] = -1*self.alpha[i]*self.alpha[j]/((a_0**2)*(a_0 + 1))
		self.covariance_matrix = covariance_matrix
		self.multivariate_mean = [0 for i in range(len(self.alpha))]

		#denominator_constant is the denominator of the f(Theta) function
		# which is 1/B(x + y) where B(x + y) can be expressed as
		# B(alpha) = ( product gamma(alpha) ) / gamma( sum (alpha) )
		# https://en.wikipedia.org/wiki/Dirichlet_distribution#Probability_density_function
		numerator = 1
		for i in range(len(self.alpha)):
			numerator *= math.gamma(self.alpha[i])
		B_x_y = numerator / math.gamma(sum(self.alpha))

		self.denominator_constant = B_x_y

		self.rejected_poins = 0
	def Metropolis(self, x_i):
		# Receive x_i current point
		# at the trajectory and returns
		# the next point of the trajectory
		# that may or may not be the same
		# based on the Metropolis acceptance
		# criterion.

		def alpha(x_i, x_j):
			return min(1, self.f(x_j)/self.f(x_i))

		def is_inside_dominion_fn(x_j):
			# n-d vector -> boolean
			# checks if n-dimensional point
			# x_j is inside Theta dominion
			if sum(x_j) > 1: return False
			for i in range(len(x_j)):
				if x_j[i] <= 0:
					return False
			return True

		# Variable declaration to start loop. If first try 
		# obeys dominion limits, it only runs once. Otherwise,
		# it runs until it gets a point inside the dominion.
		is_inside_dominion = False
		# Guarantees that x_j is inside the desired dominion
		while is_inside_dominion == False:
			y = np.random.multivariate_normal(self.multivariate_mean, self.covariance_matrix)
			x_j = [x_i[i] + y[i] for i in range(len(x_i))]
			is_inside_dominion = is_inside_dominion_fn(x_j)

		u = np.random.uniform()
		if u > alpha(x_i, x_j):
			# Reject
			x_j = x_i
		return x_j

	def generate_theta(self):
		#None -> None

		# Initial guess
		x_i = [self.alpha[i]/sum(self.alpha) for i in range(len(self.alpha))]

		# Burning in
		for i in range(10000):
			x_i = self.Metropolis(x_i)

		# Once the system is heated,
		# generate the desired n Theta observations
		thetas = [x_i]
		for i in range(self.n):
			x_j = self.Metropolis(thetas[-1])
			if x_j == thetas[-1]: self.rejected_poins += 1
			thetas.append(x_j)

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
		return (i + 1)/self.n #index divided by total n points
