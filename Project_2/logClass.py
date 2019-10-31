import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn


class LogisticRegression(object):


    def __call__(self, X):
        z = X @ self.beta
        return 1/(1 + np.exp(-z))


    def fit(self, X, y):
        pass


    def sigmoid(self, z):
        val = 1/(1 + np.exp(-z))
        return val.astype('float32')    #Memory error for float64


    def loss_function(self, y, prob):
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

        baseline = np.linspace(0, 1 + 1/len(y), len(y))

        x, best = bestCurve(defaults=defaults, total=total)

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

        def bestCurve(defaults, total):
            x = np.linspace(0, 1, total)

            y1 = np.linspace(0, 1, defaults)
            y2 = np.ones(total-defaults)
            y3 = np.concatenate([y1,y2])
            return x, y3

        x, best = bestCurve(defaults=defaults, total=total)
        yPred2 = 1 - ypred
        yPred3 = np.array((yPred2, ypred)).T
        skplt.metrics.plot_cumulative_gain(y, yPred3, title="")
        plt.plot(x, best)
        plt.savefig(filename + ".pdf")
        plt.show()


    def accuracy(self, y, ypred):
        score = (ypred.round() == y).mean()
        return score


class GradientDescent(LogisticRegression):

    def __init__(self, eta=0.01):
        self.eta = eta


    def fit(self, X, y):
        beta = np.zeros(X.shape[1])
        for i in tqdm(range(self.n)):
            z = X @ beta
            prob = self.sigmoid(z)
            gradient = np.dot(X.T, (prob - y))/y.size
            beta -= self.eta*gradient
        self.beta = beta


    def __str__(self):
        return "GRADIENT_DESCENT"

class StochasticGradient(LogisticRegression):

    def __init__(self, n_epochs=80, b_size=100):
        self.b_size = b_size
        self.n_epochs = n_epochs


    def learn_rate(self, t0, t1, t):
        return t0/(t + t1)


    def fit(self, X, y):
        beta = np.zeros(X.shape[1])
        t0, t1 = 5, 100
        for i in tqdm(range(self.n_epochs)):
            for j in range(self.b_size):
                index = np.random.randint(self.b_size)
                Xi = X[index]
                yi = y[index]
                z = Xi @ beta
                prob = self.sigmoid(z)
                gradient = (Xi.T @ (prob - yi))/yi.size
                eta = self.learn_rate(t0, t1, (i*self.b_size + j))
                beta = beta - eta*gradient
        self.beta = beta


    def __str__(self):
        return "STOCHASTIC_GRADIENT"

class StochasticGradientMiniBatch(LogisticRegression):

    def __init__(self, n_epochs=80, b_size=100, eta= 0.01):
        self.b_size = b_size
        self.n_epochs = n_epochs
        self.eta = eta


    def learn_rate(self, t0, t1, t):
        return t0/(t + t1)


    def fit(self, X, y):
        beta = np.zeros(X.shape[1])
        t0, t1 = 5, 50
        m = len(y)

        for i in tqdm(range(self.n_epochs)):
            indices = np.random.permutation(m)
            X = X[indices]
            y = y[indices]
            for j in range(0, m, self.b_size):
                Xi = X[j:j+self.b_size]
                yi = y[j:j+self.b_size]

                z = Xi @ beta
                prob = self.sigmoid(z)

                gradient = (Xi.T @ (prob - yi))/yi.size
                beta -= self.eta*gradient
        self.beta = beta


    def __str__(self):
        return "STOCHASTIC_GRADIENT_MINI_BATCH"
