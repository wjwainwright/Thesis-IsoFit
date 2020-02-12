# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def func(alpha,beta,q,N):
    def subFunc(i):
        return 1/(1+alpha*np.exp(beta*(-1*(i-q)/N)**2))
    return subFunc

x = np.arange(0,50,1)
N = len(x)
q = 35
alpha = 0.1
beta = 10
weight = func(alpha,beta,q,N)
y = [weight(a) for a in x]

plt.figure()
plt.title(fr"q={q}   $\alpha$={alpha}   $\beta$={beta}   N={N}")
plt.xlabel('Index')
plt.ylabel('Weight')
plt.scatter(x,y)