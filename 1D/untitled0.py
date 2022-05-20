#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:32:56 2022

@author: caio
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-10,11)

y=np.arange(-10,11)
y[np.abs(x-1)<1]=0

plt.plot(x,y,'ro')
