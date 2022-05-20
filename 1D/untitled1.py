#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:28:23 2022

@author: caio
"""

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,1.5*10**(15),10000)


h = 6.62*10**(-34)
c = 3*10**8
k = 1.381*10**(-23)
T = 5.8*1000
y = 8*np.pi*h*x**3 / ( (c**3) * (np.exp(h*x/(k*T))))

plt.plot(x,y,'red')

ni_max = 3.4*10**(14)
# plt.plot([ni_max,ni_max],[0,y.max()],'r--')
plt.ylabel(r'$\bar{u}\left(\frac{J}{m^3}\right)$')
plt.xlabel(r'$\nu(Hz)$')
ni_cut = 10**(15)
print(c*10**(9)/ni_cut)
# plt.plot([ni_cut,ni_cut],[0,y.max()],'g--')

ni_med = 3.8*10**(15)
# plt.plot([ni_med,ni_med],[0,y.max()],'b--')

plt.savefig('Espectro.png',dpi=100)
# plt.plot([0,x.max()],[0,0],'k--')
