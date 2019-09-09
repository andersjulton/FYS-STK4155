import numpy as np
import sys
import sklearn.linear_model as skl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Regression(object):
    def __init__(self, p, n, f, l = 0, method = ''):
        """
        Class for OLS, Ridge and LASSO regression.
        p = polynomial degree
        n = number of datapoints
        f = function for testing
        """

        self.p, self.n, self.l = p, n, l
        x, y = np.sort(np.random.uniform(0, 1, n)), np.sort(np.random.uniform(0, 1, n))
        self.xm, self.ym = np.meshgrid(x, y)
        self.x, self.y = np.ravel(self.xm), np.ravel(self.ym)
        self.zm = f(self.xm, self.ym)
        self.z = np.ravel(self.zm)
        self.CreateDesignMatrix_X()
        if method == "OLS":
            self.method = self.OLS
        elif method == "RIDGE":
            self.method = self.RIDGE
        elif self.method == "LASSO":
            self.method = self.LASSO
        else:
            print("Regression method not specified")
            sys.exit()

    def plotCompare(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(self.xm, self.ym, self.zm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def CreateDesignMatrix_X(self):
        N = len(self.x)
        l = int((self.p + 1)*(self.p + 2)/2)
        self.X = np.ones((N, l))

        for i in range(1, self.p + 1):
            q = int((i)*(i + 1)/2)
            for k in range(i + 1):
                self.X[:, q + k] = self.x**(i - k)*self.y**k

    def TTsplit(self, test_size):
        interval = np.sort(np.random.choice(len(self.z), replace = False, size = int(len(self.z)*test_size)))
        X_test, z_test = self.X[interval,:], self.z[interval]
        X_train, z_train = np.ma.array(self.X, mask = False), np.ma.array(self.z, mask = False)
        z_train.mask[interval] = True
        X_train.mask[interval,:] = True
        self.X = np.ma.compress_rows(X_train)
        self.z = z_train.compressed()

    def OLS(self):
        U, s, VT = np.linalg.svd(self.X)
        D = np.diag(s**2)
        Xinv = np.linalg.inv(VT.T.dot(D).dot(VT))
        self.beta = Xinv.dot(self.X.T).dot(self.z)
        self.z_tilde = self.X @ self.beta

    def RIDGE(self):
        U, s, VT = np.linalg.svd(self.X)
        D = np.diag(s**2)
        Xinv = np.linalg.inv(VT.T.dot(D).dot(VT))
        self.beta = Xinv.dot(self.X.T).dot(self.z)/(1 + self.l)
        self.z_tilde = self.X @ self.beta

    def LASSO(self):
        clf_lasso = skl.Lasso(alpha=self.l).fit(self.X, self.z)

    def confIntBeta(self):
        self.method()
        varbeta = np.sqrt(np.linalg.inv(self.X.T.dot(self.X)).diagonal())
        percentiles = [99, 98, 95, 90]
        z = [2.576, 2.326, 1.96, 1.645]
        sigmaSQ = np.sum((self.z - self.z_tilde)**2)/(len(self.z) - len(self.beta) - 1)
        for k in range(len(self.beta)):
            print("Confidence interval for beta %i" % (k + 1))
            for i, n in enumerate(percentiles):
                print("%2i%%: %3.2f +- %3.2f" % (percentiles[i], self.beta[k], z[i]*np.sqrt(sigmaSQ)/varbeta[k]))

    def kFoldCV(self, k = 10, shuffle = False):
        if shuffle == True:
            interval = np.random.choice(len(self.z), replace = False, size = int(len(self.z)))
            isplit = np.sort(np.array_split(interval, k))
        else:
            interval = np.arange(len(self.z))
            isplit = np.array_split(interval, k)
        kR2 = 0
        kMSE = 0
        X_train, z_train = np.ma.array(self.X, mask = False), np.ma.array(self.z, mask = False)

        for i in range(k):
            z_train.mask[isplit[i]] = True
            X_train.mask[isplit[i],:] = True
            self.X = np.ma.compress_rows(X_train)
            self.z = z_train.compressed()
            self.method()

            kR2 += self.R2()
            kMSE += self.MSE()
            z_train.mask[isplit[i]] = False
            X_train.mask[isplit[i],:] = False

        return kR2/k, kMSE/k

    def R2(self):
        return 1 - np.sum((self.z - self.z_tilde)**2)/np.sum((self.z - np.mean(self.z))**2)

    def MSE(self):
        return np.mean((self.z - self.z_tilde)**2)

    def relError(self, x, x_tilde):
        return abs((x - x_tilde)/x)


if __name__ == "__main__":

    def FrankeFunction(x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
        term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
        term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
        term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
        return term1 + term2 + term3 + term4

    p = 5
    n = 10
    l = 0.5
    f = FrankeFunction
    method = "OLS"
    a = Regression(p, n, f, l, method)
    #a.plotCompare()
