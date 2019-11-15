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
        return self.perceptron(z)


    def fit(self, X, y):
        raise Error("Method fit(X, y) not implemented")


    # sigmoid
    def perceptron(self, z):
        val = 1/(1 + np.exp(-z))
        return val.astype('float32')    #Memory error for float64


    def loss_function(self, y, prob):
        return (-y*np.log(prob) - (1 - y)*np.log(1 - prob)).mean()


    def _bestCurve(self, defaults, total):
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

        x, best = self._bestCurve(defaults, total)

        modelArea = np.trapz(lines.get_ydata(), lines.get_xdata())
        bestArea = np.trapz(best, x)
        baselineArea = np.trapz(baseline, baseline)
        ratio = (modelArea - baselineArea)/(bestArea - baselineArea)

        return ratio



    def plot(self, y, ypred, filename):
        defaults = sum(y == 1)
        total = len(y)

        x, best = self._bestCurve(defaults,total)

        x_data, y_data = skplt.helpers.cumulative_gain_curve(y, ypred)

        plt.plot(x_data, y_data, label="Model curve", color='orange')
        plt.plot(x, best, label="Best curve", color='green', linewidth=2)
        plt.plot(x, x, label="Baseline", linestyle="--", color='black')
        plt.xlabel("Percentage of sample", fontsize=14)
        plt.ylabel("Gain",fontsize=14)
        plt.grid()
        plt.legend(fontsize=14)
        plt.tick_params('both', labelsize=12)
        plt.tight_layout()
        plt.savefig(filename + ".pdf")
        plt.show()
        plt.close()


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

    def __init__(self, eta=0.001, max_iter=10000):
        self.eta = eta
        self.N = max_iter
        self._name = "GRADIENT DESCENT"


    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        for i in range(self.N):
            p = self.perceptron(X @ theta)
            gradient = np.dot(X.T, (p - y))/y.size
            theta -= self.eta*gradient
        self.theta = theta



class StochasticGradient(LogisticRegression):
    """
    Stochastic Gradient Descent method with momentum.
    """

    def __init__(self, N_epochs=100, eta=0.001, gamma=0.9):
        self.N_epochs = N_epochs
        self.gamma = gamma
        self.eta = eta
        self._name = "STOCHASTIC GRADIENT"


    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        M = len(y)
        v = 0
        for i in range(self.N_epochs):
            for j in range(M):
                index = np.random.randint(M)
                Xi = X[index]
                yi = y[index]
                pi = self.perceptron(Xi @ theta)
                gradient = np.dot(Xi.T, (pi - yi))
                v = self.gamma*v + self.eta*gradient
                theta -= v
        self.theta = theta


class StochasticGradientMiniBatch(LogisticRegression):
    """
    Stochastic Gradient Descent method with mini-batches and momentum.
    """

    def __init__(self, N_epochs=100, b_size=500, eta=0.01, gamma=0.9):
        self.b_size = b_size
        self.N_epochs = N_epochs
        self.eta = eta
        self.gamma = gamma
        self._name = "STOCHASTIC GRADIENT MINI BATCH"


    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        M = len(y)
        j_max = M - self.b_size + 1
        v = 0
        for i in range(self.N_epochs):
            indices = np.random.permutation(M)
            X = X[indices]
            y = y[indices]
            for j in range(0, j_max, self.b_size):
                Xi = X[j:j+self.b_size]
                yi = y[j:j+self.b_size]

                pi = self.perceptron(Xi @ theta)
                gradient = np.dot(Xi.T, (pi - yi))/yi.size

                v = self.gamma*v + self.eta*gradient
                theta -= v

        self.theta = theta



class ADAM(LogisticRegression):
    """
    ADAM optimizer with mini-batches.
    """

    def __init__(self, N_epochs=100, b_size=500, eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.b_size = b_size
        self.N_epochs = N_epochs
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._name = "ADAM"


    def fit(self, X, y):
        theta = np.zeros(X.shape[1])
        m, s = 0, 0
        M = len(y)
        j_max = M - self.b_size + 1
        t = 1
        for i in range(self.N_epochs):
            indices = np.random.permutation(M)
            X = X[indices]
            y = y[indices]
            for j in range(0, j_max, self.b_size):
                Xi = X[j:j+self.b_size]
                yi = y[j:j+self.b_size]

                pi = self.perceptron(Xi @ theta)
                gradient = np.dot(Xi.T, (pi - yi))

                m = self.beta1*m + (1 - self.beta1)*gradient
                s = self.beta2*s + (1 - self.beta2)*gradient**2
                mhat = m/(1 - self.beta1**t)
                shat = s/(1 - self.beta2**t)

                theta -= self.eta*mhat/(np.sqrt(shat) + self.eps)
                t += 1

        self.theta = theta
