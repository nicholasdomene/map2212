# Exercício Programa 3 - MAP2212
# Nicholas Gialluca Domene
# Número USP 8543417
# 23 de maio de 2021

import math
import numpy as np
from scipy.stats import beta
from scipy.stats.stats import pearsonr
import random
import datetime
import ghalton
random_seed = 42
sequencer = ghalton.GeneralizedHalton(1, random_seed)
'''
- As an Improvement on your 2nd Programming Exercise,
consider replacing the Pseudo Random Number Generator
by a Quasi Random Number Generator.

- Do your Monte Carlo integration routines work better?
Empirically, how faster are now your integration routines?

- You should carefully explain how and why you did
your empirical analysis and reached your conclusions.
'''

def f(x):
	cpf = 0.45361387819
	rg  = 0.384850546
	return math.exp(-rg*x)*math.cos(cpf*x)

def beta_density(x, alpha, beta):
	beta_hat = math.gamma(alpha)*math.gamma(beta)/math.gamma(alpha+beta)
	return (x**(alpha - 1)*(1 - x)**(beta-1))/beta_hat

def crude(n, generator_method="pseudo"):
	#int, str -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and a random generating method ('pseudo' or 'quadi') and 
	generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	list_f_x = []
	gamma_hat = 0
	for i in range(n):
		if generator_method=="pseudo":
			x = np.random.uniform(low=0, high=1)
		else:
			x = sequencer.get(1)[0][0]
		f_x = f(x)
		list_f_x.append(f_x)
		gamma_hat += f_x/n

	variance_gamma_hat = np.var(list_f_x)

	standard_error = math.sqrt(variance_gamma_hat/n)
	relative_error = standard_error/gamma_hat
	is_error_below_threshold = True if 1.65*relative_error < 0.0005 else False

	return gamma_hat, relative_error, is_error_below_threshold

def hit_or_miss(n, generator_method="pseudo"):
	#int -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and a random generating method ('pseudo' or 'quadi') and 
	generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	gamma_hat = 0
	for i in range(n):
		if generator_method=="pseudo":
			x = np.random.uniform(low=0, high=1)
			y = np.random.uniform(low=0, high=1)
		else:
			y = sequencer.get(1)[0][0]
			x = sequencer.get(1)[0][0]
		f_x = f(x)
		if y <= f_x:
			gamma_hat += 1

	gamma_hat = gamma_hat/n
	variance_gamma_hat = gamma_hat*(1 - gamma_hat)

	standard_error = math.sqrt(variance_gamma_hat/n)
	relative_error = standard_error/gamma_hat
	is_error_below_threshold = True if 1.65*relative_error < 0.0005 else False

	return gamma_hat, relative_error, is_error_below_threshold

def importance_sampling(n, generator_method="pseudo"):
	#int -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and a random generating method ('pseudo' or 'quadi') and 
	generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	a_beta, b_beta = 1, 1 #determined visually
	gamma_hat = 0
	gamma_hat2 = 0
	for i in range(n):
		if generator_method=="pseudo":
			x = np.random.uniform(low=0, high=1)
		else:
			x = sequencer.get(1)[0][0]
		f_x = f(x)
		f_x2 = f(x**2)
		g_x = beta_density(x, a_beta, b_beta)
		g_x2 = beta_density(x**2, a_beta, b_beta)
		gamma_hat += (f_x/g_x)/n
		gamma_hat2 += (f_x2/g_x2)/n

	variance_gamma_hat = gamma_hat2 - gamma_hat**2

	standard_error = math.sqrt(variance_gamma_hat/n)
	relative_error = standard_error/gamma_hat
	is_error_below_threshold = True if 1.65*relative_error < 0.0005 else False

	return gamma_hat, relative_error, is_error_below_threshold

def control_variate(n, generator_method="pseudo"):
	#int -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and a random generating method ('pseudo' or 'quadi') and
	generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	def g(x):
		return 1 - (2/5)*x

	#store lists to calculate correlation and variance
	list_f_x  = []
	list_g_x  = []
	gamma_hat = 0
	for i in range(n):
		if generator_method=="pseudo":
			x = np.random.uniform(low=0, high=1)
		else:
			x = sequencer.get(1)[0][0]
		f_x = f(x)
		g_x = g(x)
		list_f_x.append(f_x)
		list_g_x.append(g_x)
		gamma_hat += (f_x - g_x + 4/5)/n

	'''
	pearsonr returns two values: Pearson’s correlation coefficient
	and the 2-tailed p-value, according to the documentation
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
	'''
	rho = pearsonr(list_f_x, list_g_x)[0]

	var_f_x = np.var(list_f_x)
	stddev_f_x = math.sqrt(var_f_x)
	var_g_x = np.var(list_g_x)
	stddev_g_x = math.sqrt(var_g_x)

	variance_gamma_hat = (1/n)*(
		var_f_x + var_g_x - 2*rho*stddev_f_x*stddev_g_x
		)

	standard_error = math.sqrt(variance_gamma_hat/n)
	relative_error = standard_error/gamma_hat
	is_error_below_threshold = True if 1.65*relative_error < 0.0005 else False

	return gamma_hat, relative_error, is_error_below_threshold


def run_experiment_increasing_n(variant_implementation_function, generator_method):
	is_error_below_threshold = False
	n = 1
	while is_error_below_threshold == False:
		n *= 2
		gamma_hat, relative_error, is_error_below_threshold = variant_implementation_function(n, generator_method)

	return gamma_hat, relative_error, n


def __main__():
	random_seed = 1
	np.random.seed(random_seed)
	random_seed = 42
	sequencer = ghalton.GeneralizedHalton(1, random_seed)

	print("MAP2212")
	print("Nicholas Gialluca Domene")
	print("N USP 8543417")
	print("May 23rd 2021")
	print("Executing solution for Programming Exercise 3")
	print("Random seed: ", random_seed)
	print("According to WolframAlpha")
	print("- Integral of f(x) between 0 and 1: 0.80452")
	print()
	print()

	print("Crude Implementation")
	print("- Pseudo-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, n = run_experiment_increasing_n(crude, generator_method="pseudo")
	t1 = datetime.datetime.now()
	print("  Gamma hat:      ", gamma_hat)
	print("  Relative error: ", error)
	print("  N:              ", n)
	print("  Time taken:     ", t1-t0)
	print()
	print("- Quasi-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, n = run_experiment_increasing_n(crude, generator_method="quasi")
	t1 = datetime.datetime.now()
	print("  Gamma hat:      ", gamma_hat)
	print("  Relative error: ", error)
	print("  N:              ", n)
	print("  Time taken:     ", t1-t0)
	print()
	print()
	sequencer.reset()

	print("Hit or Miss Implementation")
	print("- Pseudo-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, is_error_below_threshold = hit_or_miss(2722500, generator_method="pseudo")
	t1 = datetime.datetime.now()
	print("  Gamma hat: ", gamma_hat)
	print("  Relative error: ", error)
	print("  Is error below threshold: ", is_error_below_threshold)
	print("  Time taken: ", t1-t0)
	print()
	print("- Quaise-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, is_error_below_threshold = hit_or_miss(2722500, generator_method="quasi")
	t1 = datetime.datetime.now()
	print("  Gamma hat: ", gamma_hat)
	print("  Relative error: ", error)
	print("  Is error below threshold: ", is_error_below_threshold)
	print("  Time taken: ", t1-t0)
	print()
	print()
	sequencer.reset()


	print("Importance Sampling Implementation")
	print("- Pseudo-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, n = run_experiment_increasing_n(importance_sampling, generator_method="pseudo")
	t1 = datetime.datetime.now()
	print("  Gamma hat:      ", gamma_hat)
	print("  Relative error: ", error)
	print("  N:              ", n)
	print("  Time taken:     ", t1-t0)
	print()
	print("- Quasi-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, n = run_experiment_increasing_n(importance_sampling, generator_method="quasi")
	t1 = datetime.datetime.now()
	print("  Gamma hat:      ", gamma_hat)
	print("  Relative error: ", error)
	print("  N:              ", n)
	print("  Time taken:     ", t1-t0)
	print()
	print()
	sequencer.reset()

	print("Control Variate Implementation")
	print("- Pseudo-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, n = run_experiment_increasing_n(control_variate, generator_method="pseudo")
	t1 = datetime.datetime.now()
	print("  Gamma hat:      ", gamma_hat)
	print("  Relative error: ", error)
	print("  N:              ", n)
	print("  Time taken:     ", t1-t0)
	print()
	print("- Quasi-random")
	t0 = datetime.datetime.now()
	gamma_hat, error, n = run_experiment_increasing_n(control_variate, generator_method="quasi")
	t1 = datetime.datetime.now()
	print("  Gamma hat:      ", gamma_hat)
	print("  Relative error: ", error)
	print("  N:              ", n)
	print("  Time taken:     ", t1-t0)
	print()
	print()

__main__()


