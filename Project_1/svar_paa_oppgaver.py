import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from franke import *
from regClass import OLS, LASSO, RIDGE

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
test, train = get_test_train_data(200, 1./k, True)

ols = OLS(5)
ztilde, MSE, R2 = ols.kFoldCV(train[0], train[1], train[2], k, len(test[-1]))

MSE = np.mean(np.mean((test[-1] - ztilde)**2, axis=1, keepdims=True))
print(MSE)
