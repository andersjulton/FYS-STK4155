import numpy as np
import sklearn.linear_model as skl
from time import time


class Regression(object):
    def __init__(self, input_data_x, input_data_y, output_data, poly_degree):
        self.p = poly_degree
        self.X = self.CreateDesignMatrix(input_data_x, input_data_y)
        self.z = output_data
        self.learn()


    def __call__(self, x, y):
        X = self.CreateDesignMatrix(x, y)
        return X @ self.beta


    def learn(self):
        pass
        

    def CreateDesignMatrix(self, x, y):
        N = len(x)
        M = int((self.p + 1)*(self.p + 2)/2)
        X =  np.ones((N, M))
        for i in range(1, self.p + 1):
            q = int( (i*i + i)/2 )
            for k in range(i + 1):
                X[:, q + k] = x**(i - k)*y**k
        return X


    def TTsplit(self, test_size):
        interval = np.sort(np.random.choice(len(self.z), replace=False, size=int(len(self.z)*test_size)))
        X_test, z_test = self.X[interval,:], self.z[interval]
        X_train, z_train = np.ma.array(self.X, mask = False), np.ma.array(self.z, mask = False)
        z_train.mask[interval] = True
        X_train.mask[interval,:] = True
        self.X = np.ma.compress_rows(X_train)
        self.z = z_train.compressed()


    def confIntBeta(self):
        varbeta = np.sqrt(np.linalg.inv(self.X.T @ self.X)).diagonal()
        percentiles = [99, 98, 95, 90]
        z = [2.576, 2.326, 1.96, 1.645]
        sigmaSQ = np.sum((self.z - self.z_tilde)**2)/(len(self.z) - len(self.beta) - 1)
        for k in range(len(self.beta)):
            print("Confidence interval for beta %i" % (k + 1))
            for i, n in enumerate(percentiles):
                print("%2i%%: %3.2f +- %3.2f" % (percentiles[i], self.beta[k], z[i]*np.sqrt(sigmaSQ)*varbeta[k]))


    def kFoldCV(self, k=10, shuffle=False):
        if shuffle:
            interval = np.random.choice(len(self.z), replace=False, size=int(len(self.z)))
            isplit = np.sort(np.array_split(interval, k))
        else:
            interval = np.arange(len(self.z))
            isplit = np.array_split(interval, k)
        kR2 = 0
        kMSE = 0
        X_train, z_train = np.ma.array(self.X, mask=False), np.ma.array(self.z, mask=False)

        for i in range(k):
            z_train.mask[isplit[i]] = True
            X_train.mask[isplit[i],:] = True
            self.X = np.ma.compress_rows(X_train)
            self.z = z_train.compressed()
            self.learn()

            kR2 += self.R2()
            kMSE += self.MSE()
            z_train.mask[isplit[i]] = False
            X_train.mask[isplit[i],:] = False

        return kR2/k, kMSE/k

    # the RR coefficient of determination.
    def RR(self):
        RR_res = np.sum((self.z - self.z_tilde)**2)
        RR_tot = np.sum((self.z - np.mean(self.z))**2)
        return 1 - RR_res/RR_tot

    
    # mean squared error 
    def MSE(self, z=None, z_tilde=None):
        if z == None or z_tilde == None:
            z = self.z 
            z_tilde = self.z_tilde
        return np.mean((z - z_tilde)**2)


    # residual sum of squares
    def RRS(self):
        return sum((self.z - self.z_tilde)**2)


    # relative error
    def relError(self):
        return abs((self.z - self.z_tilde)/self.z)





class OLS(Regression):

    def learn(self):
        # eigh finds Ax = lx for symmetric/hermitian A
        E, P = np.linalg.eigh( self.X.T@self.X )
        D_inv = np.diag(1/E)
        self.beta = P @ D_inv @ P.T @ self.X.T @ self.z
        self.z_tilde = self.X @ self.beta


    def test(self):
        # time old version
        start = time()
        U, sigma, VT = np.linalg.svd(self.X)
        D = np.diag(sigma**2)
        XTXinv = np.linalg.inv(VT.T @ D @ VT)
        self.beta = XTXinv @ self.X.T @ self.z
        self.z_tilde = self.X @ self.beta
        end = time()
        print("Old: %.3e" %float(end - start))
        score = self.RR()
        # time new version
        start = time()
        self.learn()
        end = time()
        print("New: %.3e" %float(end - start))
        # compare RR
        assert abs(score - self.RR()) < 1e-12, "Assumtion of Beta in OLS is wrong?"



class RIDGE(Regression):
    def __init__(self, input_data_x, input_data_y, output_data, poly_degree, l):
        self.l = l
        super().__init__(input_data_x, input_data_y, output_data, poly_degree)


    def learn(self):
        I = np.identity(len(self.X[0]))
        self.beta = np.linalg.inv(self.X.T @ self.X + self.l*I) @ self.X.T @ self.z
        self.z_tilde = self.X @ self.beta




class LASSO(Regression):
    def __init__(self, input_data_x, input_data_y, output_data, poly_degree, l):
        self.l = l
        super().__init__(input_data_x, input_data_y, output_data, poly_degree)


    def learn(self):
        lasso = skl.Lasso(alpha=self.l).fit(self.X, self.z)
        self.beta = lasso.coef_
        self.beta[0] = lasso.intercept_
        self.z_tilde = self.X @ self.beta 


    def test(self):
        lasso = skl.Lasso(alpha=self.l).fit(self.X, self.z)
        score = lasso.score(self.X, self.z)
        assert abs(score - self.RR()) < 1e-12, "Assumtion of Beta in Lasso is wrong?"

