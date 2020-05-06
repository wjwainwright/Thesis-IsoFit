# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#BP-RP = 27
#B-V = 55-56
input = np.genfromtxt('xmatch.csv',delimiter=',',skip_header=1)

cols = [27,55,56]

data = []

for row in input:
    dump = False
    for c in cols:
        if np.isnan(row[c]):
            dump = True
    if dump == False and row[55] < 15:
        data.append([row[b] for b in cols])

plt.figure()
x = [a[1]-a[2] for a in data]
y = [a[0] for a in data]
plt.scatter(x,y)
plt.xlabel('B-V')
plt.ylabel('BP-RP')

fit = np.polyfit(x,y,1)
plt.plot(x,[fit[0]*a+fit[1] for a in x],color='red',label=f'y={fit[0]:.2f}*x+{fit[1]:.2f}')
plt.legend()
        
            
    

