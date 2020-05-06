# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def func(alpha,beta,c,q,N):
    def subFunc(i):
        #return c/(1+alpha*np.exp(beta*((i-q)/N)**2))
        return c/(1+alpha*np.exp(beta*((i-q)/N)**4))
    return subFunc

x = np.arange(0,61,1)
N = len(x) #Number of points in the array
q = 30 #Anchor point of max weight
alpha = 0.5 #Less than 1, flatness of tail post-peak
beta = 10 #Steepness of drop off
c = alpha+1 #Scales max weight to 1
weight = func(alpha,beta,c,q,N)
y = [weight(a) for a in x]

plt.figure()
plt.title(fr"q={q}   $\alpha$={alpha}   $\beta$={beta}   c={c}   N={N}")
plt.xlabel('Index')
plt.ylabel('Weight')
plt.scatter(x,y)