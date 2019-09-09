import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model

def designMatrix(n, p, noise):
    x = np.random.rand(n,1)
    y = 5*x*x + noise*np.random.randn(n,1)
    X = np.zeros((n, p))

    for i in range(n):
        for k in range(p):
            X[i][k] = x[i]**(p)
    return x, y, X



def beta(X, y, lamb):
    beta = np.linalg.inv(X.T.dot(X) + lamb*np.identity(len(X[0]))).dot(X.T).dot(y)
    return beta


for i in range(1, 11, 1):
    print(i)
    for k in range(1, 11, 1):
        x, y, X = designMatrix(100, 3, k/10)
        betaR = beta(X, y, i/10)
        ytilde = X @ betaR
        print(r2_score(y, ytilde))
