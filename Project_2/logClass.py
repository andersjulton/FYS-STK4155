import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn


class LogisticRegression:

    def __init__(self):
        self._name = "Logistic Regression"

    def __call__(self, X):
        z = X @ self.theta
        return 1/(1 + np.exp(-z))


    def fit(self, X, y):
        pass


    def sigmoid(self, z):
        val = 1/(1 + np.exp(-z))
        return val.astype('float32')    #Memory error for float64


    def loss_function(self, y, prob):
        return (-y*np.log(prob) - (1 - y)*np.log(1 - prob)).mean()


    def bestCurve(self, defaults, total):
        x = np.linspace(0, 1, total)

        y1 = np.linspace(0, 1, defaults)
        y2 = np.ones(total-defaults)
        y = np.concatenate([y1,y2])
        return x, y


    def get_Area_ratio(self, y, ypred):
        ypred = np.array((1 - ypred, ypred)).T
        ax = skplt.metrics.plot_cumulative_gain(y, ypred)
        plt.close()
        lines = ax.lines[1]

        defaults = sum(y == 1)
        total = len(y)

        baseline = np.linspace(0, 1 + 1/total, total)

        x, best = self.bestCurve(defaults, total)

        modelArea = np.trapz(lines.get_ydata(), lines.get_xdata())
        bestArea = np.trapz(best, x)
        baselineArea = np.trapz(baseline, baseline)
        ratio = (modelArea - baselineArea)/(bestArea - baselineArea)

        return ratio



    def plot(self, y, ypred, filename):
        seaborn.set(style="white", context="notebook", font_scale=1,
                    rc={"axes.grid": True, "legend.frameon": False,
        "lines.markeredgewidth": 1.4, "lines.markersize": 10, "figure.figsize": (7, 6)})
        seaborn.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 4.5})

        defaults = sum(y == 1)
        total = len(y)

        x, best = self.bestCurve(defaults, total)
        ypred = np.array((1 - ypred, ypred)).T
        skplt.metrics.plot_cumulative_gain(y, yPred, title="")
        
        plt.plot(x, best)
        plt.savefig(filename + ".pdf")
        plt.show()


    def accuracy(self, y, ypred):
        score = (ypred.round() == y).mean()
        return score

    def __str__(self):
        return self._name.replace(" ", "_")

    def __repr__(self):
        return self._name


class GradientDescent(LogisticRegression):

    """
    Gradient Descent method
    """

    def __init__(self, eta=0.001, max_iter=100000):
        self.eta = eta
        self.n = max_iter
        self._name = "GRADIENT DESCENT"

    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        for i in tqdm(range(self.n)):
            z = X @ theta
            prob = self.sigmoid(z)
            gradient = np.dot(X.T, (prob - y))/y.size
            theta -= self.eta*gradient
        self.theta = theta



class StochasticGradient(LogisticRegression):

    """
    Stochastic Gradient Descent method with momentum.
    """

    def __init__(self, n_epochs=80, eta=0.001, gamma=0.9):
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.eta = eta
        self._name = "STOCHASTIC GRADIENT"


    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        vp = 0
        m = len(y)
        for i in tqdm(range(self.n_epochs)):
            for j in range(m):
                index = np.random.randint(m)
                Xi = X[index]
                yi = y[index]
                z = Xi @ theta
                prob = self.sigmoid(z)
                gradient = np.dot(Xi.T, (prob - yi))/yi.size
                vn = self.gamma*vp + self.eta*gradient
                theta -= vn
                vp = vn
        self.theta = theta


class StochasticGradientMiniBatch(LogisticRegression):

    """
    Stochastic Gradient Descent method with mini-batches and momentum.
    """

    def __init__(self, n_epochs=80, b_size=100, eta=0.01, gamma=0.9):
        self.b_size = b_size
        self.n_epochs = n_epochs
        self.eta = eta
        self.gamma = gamma
        self._name = "STOCHASTIC GRADIENT MINI BATCH"


    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        m = len(y)
        vp = 0
        for i in tqdm(range(self.n_epochs)):
            indices = np.random.permutation(m)
            X = X[indices]
            y = y[indices]
            for j in range(0, m, self.b_size):
                Xi = X[j:j+self.b_size]
                yi = y[j:j+self.b_size]

                z = Xi @ theta
                prob = self.sigmoid(z)

                gradient = (Xi.T @ (prob - yi))/yi.size
                vn = self.gamma*vp + self.eta*gradient
                theta -= vn
                vp = vn
        self.theta = theta



class ADAM(LogisticRegression):

    """
    ADAM optimizer with mini-batches.
    """

    def __init__(self, n_epochs=80, b_size=100, eta=0.001, beta1= 0.9, beta2=0.999, eps=1e-8):
        self.b_size = b_size
        self.n_epochs = n_epochs
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._name = "ADAM"


    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        mp = 0
        sp = 0
        m = len(y)

        for i in tqdm(range(self.n_epochs)):
            indices = np.random.permutation(m)
            X = X[indices]
            y = y[indices]
            for j in range(1, m + 1, self.b_size):
                Xi = X[j:j+self.b_size]
                yi = y[j:j+self.b_size]
                t = i*self.b_size + j/self.b_size

                z = Xi @ theta
                prob = self.sigmoid(z)
                gradient = np.dot(Xi.T, (prob - yi))/yi.size
                mn = self.beta1*mp + (1 - self.beta1)*gradient
                sn = self.beta2*sp + (1 - self.beta2)*gradient**2
                mhat = mn/(1 - self.beta1**t)
                shat = sn/(1 - self.beta2**t)
                theta -= self.eta*mhat/(np.sqrt(shat) + self.eps)
                mp = mn
                sp = sn
        self.theta = theta

