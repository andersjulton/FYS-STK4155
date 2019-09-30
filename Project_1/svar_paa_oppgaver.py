import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from franke import *
from regClass import OLS, LASSO, RIDGE
import tqdm

np.random.seed(2019)
"""
Part A

"""

def partA():
    #Testing MSE and R2 for higher order polynomial against real function

    xn, yn, zn = get_train_data(100, True)
    x, y, z = get_test_data(100, False)

    for i in range(6):
        ols = OLS(i)
        X = ols.CreateDesignMatrix(xn, yn)
        ols.fit(X, zn)
        ztilde = ols(X)
        print("Polynomial degree: %1i, MSE score: %1.5f, R2 score: %1.5f"% (i, ols.MSE(z, ztilde), ols.R2(z, ztilde)))


    #Print confidence interval with p = 5
    ols = OLS(5)
    ols.confIntBeta(xn, yn, zn)


"""
Part B
"""
k = 10
n = 110
test, train = get_test_train_data(n, 1/(k + 1), False)
ztest = test[-1]

p = np.arange(1,9)
error = np.zeros(len(p))
bias = error.copy()
variance = error.copy()

for i, deg in enumerate(tqdm.tqdm(p)):
    ols = RIDGE(deg, 0.00034)
    ztilde = ols.kFoldCV(train[0], train[1], train[2], k)

    error[i] = np.mean( np.mean((ztest - ztilde)**2, axis=1, keepdims=True) )
    bias[i] = np.mean( (ztest - np.mean(ztilde, axis=1, keepdims=True))**2 )
    variance[i] = np.mean( np.var(ztilde, axis=1, keepdims=True) )

plt.plot(p, error)
plt.plot(p, bias)
plt.plot(p, variance)
plt.show()
