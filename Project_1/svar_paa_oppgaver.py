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
test, train = get_test_train_data(200, 0.2, True)

ols = OLS(5)
X_test = ols.CreateDesignMatrix(test[0], test[1])
ols.fit(X_test, test[2])
ztilde_test = ols(X_test)
MSE_test, R2_test = ols.MSE(test[1], ztilde_test), ols.R2(test[1], ztilde_test)

MSE_train, R2_train = ols.kFoldCV(train[0], train[1], train[2], 10)

print(MSE_test, MSE_train)
#print(R2_test, R2_train)
