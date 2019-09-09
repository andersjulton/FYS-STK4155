from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

def CreateDesignMatrix_X(x, y, n = 5):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1)*(n + 2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i)*(i + 1)/2)
        for k in range(i + 1):
            X[:, q + k] = x**(i - k)*y**k
    return X

def r2Score(y, ytilde):
    return 1 - np.sum(y - ytilde)**2/np.sum(y - np.mean(y))**2

def MSE(y, ytilde):
    return np.mean((y - ytilde)**2)

def relError(y, ytilde):
    return abs((y - y_tilde)/y)

def confIntBeta(y, X):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    ytilde = X @ beta
    varbeta = np.sqrt(np.linalg.inv(X.T.dot(X)).diagonal())
    percentiles = [99, 98, 95, 90]
    z = [2.576, 2.326, 1.96, 1.645]
    sigmaSQ = np.sum((y - ytilde)**2)/(len(y) - len(beta) - 1)
    for k in range(len(beta)):
        print("Confidence interval for beta %i" % (k + 1))
        for i, n in enumerate(percentiles):
            print("%2i%%: %3.2f +- %3.2f" % (percentiles[i], beta[k], z[i]*np.sqrt(sigmaSQ)/varbeta[k]))

def TT_split(X, y, test_size = 0.2):
    interval = np.sort(np.random.choice(len(y), replace = False, size = int(len(y)*test_size)))
    X_test, y_test = X[interval,:], y[interval]
    X_train, y_train = np.ma.array(X, mask = False), np.ma.array(y, mask = False)
    y_train.mask[interval] = True
    X_train.mask[interval,:] = True
    X_train = np.ma.compress_rows(X_train)
    return X_train, X_test, y_train.compressed(), y_test

def k_fold_CV(X, y, k, shuffle = False):
    if shuffle == True:
        interval = np.random.choice(len(y), replace = False, size = int(len(y)))
        isplit = np.sort(np.array_split(interval, k))
    else:
        interval = np.arange(len(y))
        isplit = np.array_split(interval, k)
    kR2 = 0
    kMSE = 0
    for i in range(k):
        X_train, y_train = np.ma.array(X, mask = False), np.ma.array(y, mask = False)
        y_train.mask[isplit[i]] = True
        X_train.mask[isplit[i],:] = True
        X_train = np.ma.compress_rows(X_train)
        y_train = y_train.compressed()
        beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        y_tilde = X_train @ beta
        kR2 += r2_score(y_train, y_tilde)
        kMSE += MSE(y_train, y_tilde)

    return kR2/k, kMSE/k

def Ridge(beta, lambda):
    return beta/(1 + lambda)

n_x = 10   # number of points
m = 5        # degree of polynomial

# sort the random values, else your fit will go crazy
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

# use the meshgrid functionality, very useful
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

#Transform from matrices to vectors
x_1 = np.ravel(x)
y_1 = np.ravel(y)
n = int(len(x_1))
z_1 = np.ravel(z) + np.random.random(n)

print(x)
print(x_1)
# finally create the design matrix
X = CreateDesignMatrix_X(x_1, y_1, n = m)

X_train, X_test, z_train, z_test = TT_split(X, z_1, test_size=0.33)
beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)

#confIntBeta(z_train, X_train)

# and then make the prediction
z_tilde = X_train @ beta

#clf = skl.LinearRegression().fit(X, z_1)
#fity = clf.predict(X)

#print(r2_score(z_train, z_tilde))
#print(mean_squared_error(z_train, z_tilde))

"""
t1, t2 = k_fold_CV(X, z_1, 6, shuffle = True)
"""
