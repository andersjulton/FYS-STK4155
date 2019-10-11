import numpy as np
import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn


class LogisticRegression(object):
    def __init__(self, learningRate=0.01, max_iter=10000):
        self.lr = learningRate
        self.m = max_iter

    def __call__(self, X):
        z = X @ self.beta
        return self.sigmoid(z)

    def fit(self, X, y):
        pass


    def sigmoid(self, z):
        val = 1/(1 + np.exp(-z))
        return val.astype('float32')    #Memory error for float64


    def loss_function(self, prob, y):
        return (-y*np.log(prob) - (1 - y)*np.log(1 - prob)).mean()


    def get_Area_ratio(self, y, ypred):
        yPred2 = 1 - ypred
        yPred3 = np.array((yPred2, ypred)).T
        ax = skplt.metrics.plot_cumulative_gain(y, yPred3)
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

        #modelArea = np.trapz(ypred, dx = 1/ypred.size)
        modelArea = np.trapz(lines.get_ydata(), lines.get_xdata())
        bestArea = np.trapz(best, dx = 1/best.size)
        ratio = (modelArea - 0.5)/(bestArea - 0.5)
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
        skplt.metrics.plot_cumulative_gain(y, ypred)
        plt.plot(x, best)
        plt.show()

class GradientDescent(LogisticRegression):

    def fit(self, X, y):
        beta = np.zeros(X.shape[1])
        for i in tqdm.tqdm(range(self.m)):
            z = X @ beta
            prob = self.sigmoid(z)
            gradient = (X.T @ (prob - y))/y.size #Add more gradient methods
            beta -= self.lr*gradient
        self.beta = beta

    def __str__(self):
        return "GRADIENT_DESCENT"

class NewtonRaphsons(LogisticRegression):

    def fit(self, X, y):
        something = X

    def __str__(self):
        return "NEWTON_RAPHSONS"

class StochasticGradient(LogisticRegression):

    def fit(self, X, y):
        something = X

    def __str__(self):
        return "STOCHASTIC_GRADIENT"
