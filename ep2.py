# Exercício Programa 2 - MAP2212
# Nicholas Gialluca Domene
# Número USP 8543417
# 16 de maio de 2021

import math
import numpy as np
from scipy.stats import beta
from scipy.stats.stats import pearsonr
import random
import datetime

'''
- Find out how to use, in your computational environment, 
library functions to generate random variables with Uniform, 
Beta, Gamma and Weibull distributions.

- Implement the four variants of Monte Carlo integration 
we studied to integrate the function 
$f(x) = e^{-ax}cos(bx)$ in $[0, 1]$, where $a = 0.RG$, $b = 0.CPF$, 
and RG and CPF stand for digits of your official IDs.

- Choose parameters for each sampling distribution by visual 
inspection or any other method you like (adapt domain?). 
Choose a polynomial function for control variate. 
Choose $n$ to get a relative error 
$\frac{|\hat\gamma - \gamma |}{\gamma} < 0.0005$ 
(without knowing $\gamma$!)
'''

def f(x):
	cpf = 0.45361387819
	rg  = 0.384850546
	return math.exp(-rg*x)*math.cos(cpf*x)

def beta_density(x, alpha, beta):
	beta_hat = math.gamma(alpha)*math.gamma(beta)/math.gamma(alpha+beta)
	return (x**(alpha - 1)*(1 - x)**(beta-1))/beta_hat

def crude(n):
	#int -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	list_f_x = []
	gamma_hat = 0
	for i in range(n):
		x = np.random.uniform(low=0, high=1)
		f_x = f(x)
		list_f_x.append(f_x)
		gamma_hat += f_x/n

	variance_gamma_hat = np.var(list_f_x)

	standard_error = math.sqrt(variance_gamma_hat/n)
	is_error_below_threshold = True if (1.65*standard_error)/gamma_hat < 0.0005 else False

	return gamma_hat, is_error_below_threshold

def hit_or_miss(n):
	#int -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	gamma_hat = 0
	for i in range(n):
		x = np.random.uniform(low=0, high=1)
		y = np.random.uniform(low=0, high=1)
		f_x = f(x)
		if y <= f_x:
			gamma_hat += 1

	gamma_hat = gamma_hat/n
	variance_gamma_hat = gamma_hat*(1 - gamma_hat)

	standard_error = math.sqrt(variance_gamma_hat/n)
	is_error_below_threshold = True if (1.65*standard_error)/gamma_hat < 0.0005 else False

	return gamma_hat, is_error_below_threshold

def importance_sampling(n):
	#int -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	a_beta, b_beta = 1, 1 #determined visually
	gamma_hat = 0
	gamma_hat2 = 0
	for i in range(n):
		x = np.random.beta(a=a_beta, b=b_beta)
		f_x = f(x)
		f_x2 = f(x**2)
		g_x = beta_density(x, a_beta, b_beta)
		g_x2 = beta_density(x**2, a_beta, b_beta)
		gamma_hat += (f_x/g_x)/n
		gamma_hat2 += (f_x2/g_x2)/n

	variance_gamma_hat = gamma_hat2 - gamma_hat**2

	standard_error = math.sqrt(variance_gamma_hat/n)
	is_error_below_threshold = True if (1.65*standard_error)/gamma_hat < 0.0005 else False

	return gamma_hat, is_error_below_threshold

def control_variate(n):
	#int -> float (gamma hat), boolean (is relative error below 0.0005)
	'''
	Receives an integer n that will be used to generate n points
	and generate the value of gamma hat, estimation for the integral
	of f(x) in the interval [0, 1], and its variance
	'''
	def g(x):
		return 1 - (2/5)*x

	#store lists to calculate correlation and variance
	list_f_x  = []
	list_g_x  = []
	gamma_hat = 0
	for i in range(n):
		x   = np.random.uniform(low=0, high=1)
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
	is_error_below_threshold = True if (1.65*standard_error)/gamma_hat < 0.0005 else False

	return gamma_hat, is_error_below_threshold


def run_experiment_increasing_n(variant_implementation_function):
	is_error_below_threshold = False
	n = 1
	while is_error_below_threshold == False:
		n *= 2
		gamma_hat, is_error_below_threshold = variant_implementation_function(n)

	return gamma_hat, n


def __main__():
	random_seed = 1
	np.random.seed(random_seed)
	print("MAP2212")
	print("Nicholas Gialluca Domene")
	print("N USP 8543417")
	print("May 16th 2021")
	print("Executing solution for Programming Exercise 2")
	print("Random seed: ", random_seed)
	print()

	print("Crude Implementation")
	t0 = datetime.datetime.now()
	gamma_hat, n = run_experiment_increasing_n(crude)
	t1 = datetime.datetime.now()
	print("Gamma hat: ", gamma_hat)
	print("N:         ", n)
	print("Time taken:", t1-t0)
	print()

	print("Hit or Miss Implementation")
	t0 = datetime.datetime.now()
	gamma_hat, is_error_below_threshold = hit_or_miss(2722500)
	t1 = datetime.datetime.now()
	print("Gamma hat: ", gamma_hat)
	print("Is error below threshold: ", is_error_below_threshold)
	print("Time taken: ", t1-t0)
	print()

	print("Importance Sampling Implementation")
	t0 = datetime.datetime.now()
	gamma_hat, n = run_experiment_increasing_n(importance_sampling)
	t1 = datetime.datetime.now()
	print("Gamma hat: ", gamma_hat)
	print("N:         ", n)
	print("Time taken:", t1-t0)
	print()

	print("Control Variate Implementation")
	t0 = datetime.datetime.now()
	gamma_hat, n = run_experiment_increasing_n(control_variate)
	t1 = datetime.datetime.now()
	print("Gamma hat: ", gamma_hat)
	print("N:         ", n)
	print("Time taken:", t1-t0)
	print()

__main__()


