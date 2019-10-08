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

    def confIntBeta(self, Xtest, Xtrain, ztest, ztrain, alpha = 1.96):
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



    # k-fold cross validation
    def kFoldCV(self, x, y, z, k):
        N = len(x)
        R2 = 0; MSE = 0;
        # shuffled array of indices
        indices = np.linspace(0, N-1, N)
        np.random.shuffle(indices)
        X = self.CreateDesignMatrix(x, y)

        size = N//k         # size of each interval
        mod = N % k         # in case k is not a factor in N
        end = 0
        MSEout = np.zeros(k)

        for i in range(k):
            start = end
            end += size + (1 if i < mod else 0)
            test = np.logical_and(indices >= start, indices < end)  # small part is test
            train = test == False                                   # rest is train

            self.fit(X[train], z[train])
            ztilde = self(X[test])

            MSEout[i] = self.MSE(z[test], ztilde)

            R2 += self.R2(z[test], ztilde)
            MSE += MSEout[i]

        return R2/k, MSE/k#, MSEout


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
        U, s, VT = np.linalg.svd(X, full_matrices=False)
        Dinv = np.diag(1/s)
        self.beta = VT.T @ Dinv @ U.T @ z

    def confIntBeta(self, Xtest, Xtrain, ztest, ztrain, alpha = 1.96):
        self.fit(Xtrain, ztrain)
        ztilde = self(Xtest)
        U, s, VT = np.linalg.svd(Xtrain, full_matrices=False)
        Dinv = np.diag(1/s**2)
        v = (VT.T @ Dinv @ VT).diagonal()
        zSTD = np.sum((ztest - ztilde)**2)/(len(ztest) - len(self.beta) - 1)
        confint = alpha*np.sqrt(v*zSTD)

        return confint, np.sqrt(v*zSTD)

    def __str__(self):
        return "OLS"



class RIDGE(Regression):
    def __init__(self, poly_degree, l):
        self.l = l
        super().__init__(poly_degree)


    def fit(self, X, z):
        I = np.identity(len(X[0]))
        self.beta = np.linalg.inv(X.T @ X + self.l*I) @ X.T @ z

    def confIntBeta(self, Xtest, Xtrain, ztest, ztrain, alpha = 1.96):
        self.fit(Xtrain, ztrain)
        ztilde = self(Xtest)
        I = np.identity(len(Xtrain[0]))
        zSTD = np.sum((ztest - ztilde)**2)/(len(ztest) - len(self.beta) - 1)
        term1 = np.linalg.inv(Xtrain.T @ Xtrain + self.l*I)
        term2 = Xtrain.T @ Xtrain
        term3 = np.linalg.inv(Xtrain.T @ Xtrain + self.l*I)
        v = (term1 @ term2 @ term3).diagonal()
        confint = alpha*np.sqrt(v*zSTD)

        return confint, np.sqrt(v*zSTD)

    def __str__(self):
        return "RIDGE"



class LASSO(Regression):
    def __init__(self, poly_degree, l):
        self.l = l
        super().__init__(poly_degree)


    def fit(self, X, z):
        lasso = skl.Lasso(alpha=self.l, precompute=True).fit(X[:,1:], z)
        self.beta = np.zeros(len(X[0]))
        self.beta[1:] = lasso.coef_
        self.beta[0] = lasso.intercept_

    def __str__(self):
        return "LASSO"
