# Exercício Programa 4 - MAP2212
# Nicholas Gialluca Domene
# Número USP 8543417
# Felipe de Moura Ferreira
# Número USP 9864702
# 8 de julho de 2021

from ep5_class import EP5
import time
'''
PROBLEM STATEMENT TAKEN FROM https://www.youtube.com/watch?v=G5fGaT0RpYo

- Re-submit your 4th Programming exercise,

- Replacing the random sampler you have used, namely, the exact sampler for the Dirichlet distribution based on an Acceptance-Rejection generator for the Gamma distribution...

- by a Markov Chain Monte Carlo sampler based on the knowledge of just a potential for the Dirichlet distribution.

- Use as kernel for the MCMC a Multivariate Normal N(0, $\Sigma$) where the covariance matrix is ajusted based on your prior knowledge of the parameters of the Dirichlet andor some initial or trial sampling sequences.
'''

#Insert x and y vector here
print("Initiating...")
t1 = time.time()
x = [4, 6, 4]
y = [1, 2, 3]

test_cases = [0, 1, 0.5, 15, 20]

ep5 = EP5(x, y)

ep5.generate_theta()

print("Acceptance ratio: ", round(1 - ep5.rejected_poins/ep5.n, 4))

ep5.order_f_thetas()

for test in test_cases:
	print("U(%s) = "%(test), ep5.U(test))

t2 = time.time()
print("Time taken: ", t2-t1, " seconds")


