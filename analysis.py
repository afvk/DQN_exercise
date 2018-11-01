#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:25:09 2018

@author: afvk
"""

import pickle

import matplotlib.pyplot as plt
plt.style.use('thesis')


results = pickle.load(open('save/results.p', 'rb'))

episodes = results['episodes']
times = results['times']

plt.figure()
plt.plot(episodes, times)
plt.show()






