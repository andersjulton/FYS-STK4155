import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model

def designMatrix(n, p, noise):
    x = np.random.rand(n,1)
    y = 5*x*x + noise*np.random.randn(n,1)
    X = np.zeros((100,p))

    for i in range(100):
        for k in range(p):
            X[i][k] = x[i]**(k)
    return x, y, X

x, y, X = designMatrix(100, 3, 0.1)

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
ytilde = X @ beta
epsilon = y - ytilde

reg = linear_model.LinearRegression()
reg.fit(x, y)
ypred = reg.predict(x)

mse = mean_squared_error(y, ypred)
r2 = r2_score(y, ypred)

print(r2_score(y, ytilde), r2)
#print(r2_score(y, ytilde))
