#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code of HW4, Problem 4
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import linalg
import math

myfile = open('hw4_problem4_data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test']

predictive_mean = np.empty(X_test.shape[0],dtype=np.float64)
predictive_std = np.empty(X_test.shape[0])

sigma = 0.1
sigma_f = 1.0
ls = 3


#-------- Your code (~10 lines) ---------
N = X_train.shape[0]
M = X_test.shape[0]
kkk=sigma*sigma+sigma_f**2
knn=np.ndarray(shape=(N,N))
for i in range(N):
    for j in range(N):
        if(i!=j):
            knn[i][j] = sigma_f*sigma_f*math.exp(-(X_train[i][0]-X_train[j][0])**2/(2*(ls**2)))
        else:
            knn[i][j] = sigma**2+sigma_f**2
identic = np.identity(N)
kkn = np.ndarray(shape=(1,N))
knk = np.transpose(kkn)
for k in range (M):
    for i in range(N):
        knk[i][0] = sigma_f*sigma_f*math.exp(-(X_test[k][0]-X_train[i][0])**2/(2*(ls**2)))
    kkn = np.transpose(knk)
    predictive_mean[k] = np.matmul(np.matmul(kkn,linalg.inv(np.add(knn,(sigma**2)*identic))),Y_train)[0]
    predictive_std[k] = math.sqrt(kkk - np.matmul(np.matmul(kkn,linalg.inv(np.add(knn,(sigma**2)*identic))),knk)[0])
#---------- End of your code -----------

# Optional: Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=5, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
