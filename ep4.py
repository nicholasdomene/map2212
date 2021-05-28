# Exercício Programa 4 - MAP2212
# Nicholas Gialluca Domene
# Número USP 8543417
# Felipe de Moura Ferreira
# Número USP 9864702
# xx de maio de 2021

import math
import numpy as np
from scipy.stats import beta
import random
import datetime
random_seed = 1
np.random.seed(random_seed)
'''
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