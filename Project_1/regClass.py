import numpy as np
import sklearn.linear_model as skl


class Regression(object):
    def __init__(self, poly_degree):
        self.p = poly_degree


    def __call__(self, x, y=None):
        if y == None:
            X = x
        else:
            X = self.CreateDesignMatrix(x, y)
        return X @ self.beta

    # abstract method for regression
    def fit(self, X, z):
        pass


    def train(self, x, y, z):
        X = self.CreateDesignMatrix(x, y)
        self.fit(X, z)


    def CreateDesignMatrix(self, x, y):
        N = len(x)
        M = int((self.p + 1)*(self.p + 2)/2)
        X =  np.ones((N, M))
        for i in range(1, self.p + 1):
            q = int( (i*i + i)/2 )
            for k in range(i + 1):
                X[:, q + k] = x**(i - k)*y**k
        return X


    def confIntBeta(self, x, y, z):
        X = self.CreateDesignMatrix(x, y)
        self.fit(X, z)
        ztilde = self(X)
        E, P = np.linalg.eigh(X.T @ X)
        D_inv = np.diag(1/E)
        varbeta = np.sqrt(P @ D_inv @ P.T).diagonal()
        zSTD = np.sum((z - ztilde)**2)/(len(z) - len(self.beta) - 1)
        betaSTD = np.sqrt(zSTD)*varbeta
        percentiles = [99, 98, 95, 90]
        alpha = [2.576, 2.326, 1.96, 1.645]
        '''
        for k in range(len(self.beta)):
            print("Confidence interval for beta %i" % (k + 1))
            for i, n in enumerate(percentiles):
                print("%2i%%: %3.2f +- %3.2f" % (percentiles[i], self.beta[k], alpha[i]*betaSTD[k]))
        '''
        return betaSTD*alpha[0]


    # k-fold cross validation
    def kFoldCV(self, x, y, z, k):
        N = len(x)
        R2 = 0;    MSE = 0
        # shuffled array of indices
        indices = np.linspace(0, N-1, N)
        np.random.shuffle(indices)
        X = self.CreateDesignMatrix(x, y)

        size = N//k         # size of each interval
        mod = N % k         # in case k is not a factor in N
        end = 0
        for i in range(k):
            start = end
            end += size + (1 if i < mod else 0)
            test = np.logical_and(indices >= start, indices < end)  # small part is test
            train = test == False                                   # rest is train

            self.fit(X[train], z[train])
            z_tilde = self(X[test])

            R2 += self.R2(z[test], z_tilde)
            MSE += self.MSE(z[test], z_tilde)

        return R2/k, MSE/k


    # the RR coefficient of determination.
    def R2(self, z, z_tilde):
        RR_res = np.sum((z - z_tilde)**2)
        RR_tot = np.sum((z - np.mean(z))**2)
        return 1 - RR_res/RR_tot

    # mean squared error
    def MSE(self, z, z_tilde):
        return np.mean((z - z_tilde)**2)


    # residual sum of squares
    def RRS(self, z, z_tilde):
        return sum((z - z_tilde)**2)


    # relative error
    def relError(self, z, z_tilde):
        return abs((z - z_tilde)/z)






class OLS(Regression):

    def fit(self, X, z):
        # eigh finds Ax = lx for symmetric/hermitian A
        E, P = np.linalg.eigh(X.T @ X)
        D_inv = np.diag(1/E)
        self.beta = P @ D_inv @ P.T @ X.T @ z

    def __str__(self):
        return "OLS"



class RIDGE(Regression):
    def __init__(self, poly_degree, l):
        self.l = l
        super().__init__(poly_degree)


    def fit(self, X, z):
        I = np.identity(len(X[0]))
        self.beta = np.linalg.inv(X.T @ X + self.l*I) @ X.T @ z

    def __str__(self):
        return "RIDGE"



class LASSO(Regression):
    def __init__(self, poly_degree, l):
        self.l = l
        super().__init__(poly_degree)


    def fit(self, X, z):
        lasso = skl.Lasso(alpha=self.l).fit(X, z)
        self.beta = lasso.coef_
        self.beta[0] = lasso.intercept_

    def __str__(self):
        return "LASSO"
