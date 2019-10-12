import numpy as np
import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn


class LogisticRegression(object):
    def __init__(self, learningRate=0.01, max_iter=10000):
        self.lr = learningRate
        self.n = max_iter

    def __call__(self, X):
        z = X @ self.beta
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        pass

    def sigmoid(self, z):
        val = 1/(1 + np.exp(-z))
        return val.astype('float32')    #Memory error for float64


    def loss_function(self, prob, y):
        return (-y*np.log(prob) - (1 - y)*np.log(1 - prob)).mean()


    def get_Area_ratio(self, y, ypred):
        ypred2 = 1 - ypred
        ypred3 = np.array((ypred2, ypred)).T
        ax = skplt.metrics.plot_cumulative_gain(y, ypred3)
        plt.close()
        lines = ax.lines[1]

        defaults = sum(y == 1)
        total = len(y)

        def bestCurve(defaults, total):
            x = np.linspace(0, 1, total)

            y1 = np.linspace(0, 1, defaults)
            y2 = np.ones(total-defaults)
            y3 = np.concatenate([y1,y2])
            return x, y3

        x, best = bestCurve(defaults=defaults, total=total)

        modelArea = np.sum(lines.get_ydata()[0:-1] - x)
        bestArea = np.sum(best - x)
        ratio = modelArea/bestArea

        return ratio


    def plot(self, y, ypred):

        seaborn.set(style="white", context="notebook", font_scale=1.5,
                    rc={"axes.grid": True, "legend.frameon": False,
        "lines.markeredgewidth": 1.4, "lines.markersize": 10})
        seaborn.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})

        defaults = sum(y == 1)
        total = len(y)

        def bestCurve(defaults, total):
            x = np.linspace(0, 1, total)

            y1 = np.linspace(0, 1, defaults)
            y2 = np.ones(total-defaults)
            y3 = np.concatenate([y1,y2])
            return x, y3

        x, best = bestCurve(defaults=defaults, total=total)
        yPred2 = 1 - ypred
        yPred3 = np.array((yPred2, ypred)).T
        skplt.metrics.plot_cumulative_gain(y, yPred3)
        plt.plot(x, best)
        plt.show()

    def accuracy(self, y, ypred):
        score = (ypred.round() == y).mean()
        return score





class GradientDescent(LogisticRegression):

    def fit(self, X, y):
        beta = np.zeros(X.shape[1])
        for i in tqdm.tqdm(range(self.n)):
            z = X @ beta
            prob = self.sigmoid(z)
            gradient = np.dot(X.T, (prob - y))/y.size
            beta -= self.lr*gradient
        self.beta = beta

    def __str__(self):
        return "GRADIENT_DESCENT"

class NewtonRaphsons(LogisticRegression):

    """
    Struggles with large data set. Memory error.
    Also struggles with W containing zeros such that X.T @ W @ X is singular.
    Might be linalg solutions. SVD too slow.
    """

    def fit(self, X, y):
        beta = np.zeros(X.shape[1])
        for i in tqdm.tqdm(range(self.n)):
            z = X @ beta
            prob = self.sigmoid(z)
            W = np.diag((prob*(1 - prob)))
            Hinv = np.linalg.inv(X.T @ W @ X)
            beta -= Hinv @ -X.T @ (y - prob)
        self.beta = beta

    def __str__(self):
        return "NEWTON_RAPHSONS"

class StochasticGradient(LogisticRegression):

    def __init__(self, n_epochs = 80, max_iter=10000):
        self.m = max_iter
        self.n_epochs = n_epochs
        super().__init__(max_iter, max_iter)


    def learn_rate(self, t0, t1, t):
        return t0/(t + t1)

    def fit(self, X, y):
        beta = np.zeros(X.shape[1])
        t0, t1 = 5, 50
        for i in tqdm.tqdm(range(self.n_epochs)):
            for j in range(self.m):
                index = np.random.randint(self.m)
                Xi = X[index:index+1]
                yi = y[index:index+1]
                z = Xi @ beta
                prob = self.sigmoid(z)
                gradient = (Xi.T @ (prob - yi))/yi.size
                eta = self.learn_rate(t0, t1, (i*self.m + j))
                beta = beta - eta*gradient
        self.beta = beta

    def __str__(self):
        return "STOCHASTIC_GRADIENT"
