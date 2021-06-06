# Exercício Programa 4 - MAP2212
# Nicholas Gialluca Domene
# Número USP 8543417
# Felipe de Moura Ferreira
# Número USP 9864702
# 5 de junho de 2021

from ep4_class import EP4
import time
'''
PROBLEM STATEMENT TAKEN FROM https://www.youtube.com/watch?v=ZmaYDTLR8yw

- Consider the m-dimensional Multinomial statistical model,
with observations, x, prior information, y, and parameter Theta;

- x, y E N^m, Theta E Sm = {Theta E R^+m | Theta'1 = 1}, m = 3;

The statistical model comprises:

- Posterior density potential, 
f(Theta|x, y) = 1/B(x + y) * PI_i=1^m Theta_i^{x_i + y_i - 1}

- Cut-off set, T(v) = {Theta E Sm | f(Theta|x, y) <= v}, v >= 0

- Truth function, W(v) = integral_T(v) f(Theta|x, y) dTheta

W(v) is the posterior probability mass inside T(v), that is, 
the probability mass where the posterior potential, f(Theta|x, y),
does not exceed the threshold level v

- Dirichlet(Theta|a) = PI_{i=1}^m Theta_i^{a_i - 1} / B(a), 
B(a) = PI_{i=1}^m GAMMA(a_i) / GAMMA(\sum_{i=1}^m a_i)
B(a) is the multivariate Beta function

- Set k cut-off points, 0 = v_0 < v_1 < v_2 < ... < v_k = sup f(Theta)

- Use a Gamma RNG to generate n points in Sm,
Theta_1, ..., Theta_n, distributed according to
the posterior density function

- Use the fraction of simulated points, Theta_t, inside each
"bin", v_{j-1} <= f(Theta_t) < v_j, as an approximation of 
W(v_j) - W(v_j - 1)

- Dynamically adjust the bin's borders, v_j, to get bins of approximately
equal weight, i.e., W(t_j) - W(t_j - 1) ~= 1/k
'''

#Insert x and y vector here
print("Initiating...")
t1 = time.time()
x = [1, 1, 1]
y = [1, 1, 1]

test_cases = [0, 1, 0.9, 0.00001]

ep4 = EP4(x, y)

ep4.generate_theta()

ep4.order_f_thetas()

for test in test_cases:
	print("U(%s) = "%(test), ep4.U(test))

t2 = time.time()
print("Time taken: ", t2-t1, " seconds")

# print(ep4.min_f)

# t1 = time.time()
# print(ep4.U_obsolete(0))
# print(ep4.U_obsolete(0.00001))
# print(ep4.U_obsolete(0.9))
# t2 = time.time()
# print("Time to run: ",t2 - t1, " seconds")
# print()
# t3 = time.time()
# print(ep4.U(0))
# print(ep4.U(0.00001))
# print(ep4.U(0.9))
# t4 = time.time()
# print("Time to run: ",t4 - t3, " seconds")

# print("Time difference: ", (t4 - t3)/(t2 - t1))

# print(ep4.sup_f)

