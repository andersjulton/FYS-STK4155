import numpy as np
import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt


class NeuralNetwork(object):
    def __init__(self, X, y, n_hid_neur=50, n_cat=2, n_epochs=100, b_size=100, eta=0.1, lmbd=0.1, hid_layers=1):

        self.X_full = X
        self.Y_full = y

        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_hid_neur = n_hid_neur
        self.n_cat = n_cat
        self.hid_layers = hid_layers

        self.n_epochs = n_epochs
        self.b_size = b_size
        self.iterations = self.n_inputs // self.b_size
        self.eta = eta
        self.lmbd = lmbd

        self.get_B_W()

    def get_B_W(self):
        self.hid_W_first = np.random.randn(self.n_features, self.n_hid_neur)
        self.hid_B_first = np.zeros(self.n_hid_neur) + 0.01
        if self.hid_layers > 1:
            self.hid_W_inner = np.zeros((self.hid_layers -1, self.n_hid_neur, self.n_hid_neur))
            for i in range(self.hid_layers - 1):
                self.hid_W_inner[i] = np.random.randn(self.n_hid_neur, self.n_hid_neur)
            self.hid_B_inner = np.zeros((self.hid_layers - 1, self.n_hid_neur)) + 0.01

        self.out_W = np.random.randn(self.n_hid_neur, self.n_cat)
        self.out_B = np.zeros(self.n_cat) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.zh = self.X_part @ self.hid_W_first + self.hid_B_first
        self.ah = np.zeros((self.hid_layers, self.b_size, self.n_hid_neur))
        self.ah[0] = self.sigmoid(self.zh)
        zh_next = 0

        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = self.ah[i] @ self.hid_W_inner[i] + self.hid_B_inner[i]
                self.ah[i+1] = self.sigmoid(zh_next)
        self.zo = self.ah[-1] @ self.out_W + self.out_B
        self.probs = self.sigmoid(self.zo)

    def feed_forward_out(self, X):
        zh = X @ self.hid_W_first + self.hid_B_first
        ah = self.sigmoid(zh)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = ah @ self.hid_W_inner[i] + self.hid_B_inner[i]
                ah = self.sigmoid(zh_next)
        zo = ah @ self.out_W + self.out_B
        probs = self.sigmoid(zo)

        return probs

    def backpropagation(self):
        error_o = self.probs - self.Y_part
        error_h_prev = np.multiply(np.multiply((error_o @ self.out_W.T), self.ah[-1]), (1 - self.ah[-1]))
        self.out_W_grad = self.ah[-1].T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 2, 0, -1):
                error_h_next = np.multiply(np.multiply((error_h_prev @ self.hid_W_inner[i].T), self.ah[i]), (1 - self.ah[i]))
                hid_W_grad = self.ah[i].T @ error_h_next
                hid_B_grad = np.sum(np.asarray(error_h_next), axis=0)

                if self.lmbd > 0.0:
                    hid_W_grad += self.lmbd*self.hid_W_inner[i]
                self.hid_W_inner[i] -= self.eta*hid_W_grad
                self.hid_B_inner[i] -= self.eta*hid_B_grad
                error_h_prev = error_h_next

        hid_W_grad = self.X_part.T @ error_h_prev
        hid_B_grad = np.sum(np.asarray(error_h_prev), axis=0)

        if self.lmbd > 0.0:
            self.out_W_grad += self.lmbd*self.out_W
            hid_W_grad += self.lmbd*self.hid_W_first

        self.out_W -= self.eta*self.out_W_grad
        self.out_B -= self.eta*self.out_B_grad
        self.hid_W_first -= self.eta*hid_W_grad
        self.hid_B_first -= self.eta*hid_B_grad

    def predict(self, X):
        probs = self.feed_forward_out(X)
        return np.argmax(probs, axis=1)

    def predict_probabilities(self, X):
        probs = self.feed_forward_out(X)
        return probs

    #Add mode activation functions
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def ReLU(self, z):
        zn = np.zeros(z.shape)
        indices = np.where(z >= 0)
        zn[indices] = z[indices]
        return zn

    def train(self):
        indices = np.arange(self.n_inputs)

        for i in range(self.n_epochs):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(indices, size=self.b_size, replace=False)

                self.X_part = self.X_full[chosen_indices]
                self.Y_part = self.Y_full[chosen_indices]

                self.feed_forward()
                self.backpropagation()
            t = np.argmax(self.probs, axis=1)
            #plt.hist(t)
            #plt.show()
            #input()

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

class NeuralLogReg(NeuralNetwork):

    def feed_forward(self):
        # feed-forward for training
        self.zh = self.X_part @ self.hid_W_first + self.hid_B_first
        self.ah = np.zeros((self.hid_layers, self.b_size, self.n_hid_neur))
        self.ah[0] = self.sigmoid(self.zh)
        zh_next = 0

        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = self.ah[i] @ self.hid_W_inner[i] + self.hid_B_inner[i]
                self.ah[i+1] = self.sigmoid(zh_next)
        self.zo = self.ah[-1] @ self.out_W + self.out_B
        self.probs = self.sigmoid(self.zo)

    def feed_forward_out(self, X):
        zh = X @ self.hid_W_first + self.hid_B_first
        ah = self.sigmoid(zh)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = ah @ self.hid_W_inner[i] + self.hid_B_inner[i]
                ah = self.sigmoid(zh_next)
        zo = ah @ self.out_W + self.out_B
        probs = self.sigmoid(zo)

        return probs



    def backpropagation(self):
        error_o = self.probs - self.Y_part
        error_h_prev = np.multiply(np.multiply((error_o @ self.out_W.T), self.ah[-1]), (1 - self.ah[-1]))
        self.out_W_grad = self.ah[-1].T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 2, 0, -1):
                error_h_next = np.multiply(np.multiply((error_h_prev @ self.hid_W_inner[i].T), self.ah[i]), (1 - self.ah[i]))
                hid_W_grad = self.ah[i].T @ error_h_next
                hid_B_grad = np.sum(np.asarray(error_h_next), axis=0)

                if self.lmbd > 0.0:
                    hid_W_grad += self.lmbd*self.hid_W_inner[i]
                self.hid_W_inner[i] -= self.eta*hid_W_grad
                self.hid_B_inner[i] -= self.eta*hid_B_grad
                error_h_prev = error_h_next

        hid_W_grad = self.X_part.T @ error_h_prev
        hid_B_grad = np.sum(np.asarray(error_h_prev), axis=0)

        if self.lmbd > 0.0:
            self.out_W_grad += self.lmbd*self.out_W
            hid_W_grad += self.lmbd*self.hid_W_first

        self.out_W -= self.eta*self.out_W_grad
        self.out_B -= self.eta*self.out_B_grad
        self.hid_W_first -= self.eta*hid_W_grad
        self.hid_B_first -= self.eta*hid_B_grad

    def predict(self, X):
        probs = self.feed_forward_out(X)
        return np.argmax(probs, axis=1)

    def predict_probabilities(self, X):
        probs = self.feed_forward_out(X)
        return probs

    def train(self):
        indices = np.arange(self.n_inputs)

        for i in tqdm.tqdm(range(self.n_epochs)):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(indices, size=self.b_size, replace=False)

                self.X_part = self.X_full[chosen_indices]
                self.Y_part = self.Y_full[chosen_indices]

                self.feed_forward()
                self.backpropagation()


class NeuralLinReg(NeuralNetwork):

    def feed_forward(self):
        self.zh = self.X_part @ self.hid_W_first + self.hid_B_first
        self.ah = np.zeros((self.hid_layers, self.b_size, self.n_hid_neur))
        self.ah[0] = self.sigmoid(self.zh)
        zh_next = 0

        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = self.ah[i] @ self.hid_W_inner[i] + self.hid_B_inner[i]
                self.ah[i+1] = self.sigmoid(zh_next)
        self.zo = self.ah[-1] @ self.out_W + self.out_B
        self.ao = self.sigmoid(self.zo)

    def feed_forward_out(self, X):
        zh = X @ self.hid_W_first + self.hid_B_first
        ah = self.sigmoid(zh)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = ah @ self.hid_W_inner[i] + self.hid_B_inner[i]
                ah = self.sigmoid(zh_next)
        zo = ah @ self.out_W + self.out_B
        ao = self.sigmoid(zo)

        return ao

    def backpropagation(self):
        error_o = 2*(self.ao - self.Y_part)/len(self.ao)
        error_h_prev = np.multiply(np.multiply((error_o @ self.out_W.T), self.ah[-1]), (1 - self.ah[-1]))

        self.out_W_grad = self.ah[-1].T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 2, 0, -1):
                error_h_next = np.multiply(np.multiply((error_h_prev @ self.hid_W_inner[i].T), self.ah[i]), (1 - self.ah[i]))
                hid_W_grad = self.ah[i].T @ error_h_next
                hid_B_grad = np.sum(np.asarray(error_h_next), axis=0)

                if self.lmbd > 0.0:
                    hid_W_grad += self.lmbd*self.hid_W_inner[i]
                self.hid_W_inner[i] -= self.eta*hid_W_grad
                self.hid_B_inner[i] -= self.eta*hid_B_grad
                error_h_prev = error_h_next

        hid_W_grad = self.X_part.T @ error_h_prev
        hid_B_grad = np.sum(np.asarray(error_h_prev), axis=0)

        if self.lmbd > 0.0:
            self.out_W_grad += self.lmbd*self.out_W
            hid_W_grad += self.lmbd*self.hid_W_first

        self.out_W -= self.eta*self.out_W_grad
        self.out_B -= self.eta*self.out_B_grad
        self.hid_W_first -= self.eta*hid_W_grad
        self.hid_B_first -= self.eta*hid_B_grad

    def predict(self, X):
        z = self.feed_forward_out(X)
        return z


    def train(self):
        indices = np.arange(self.n_inputs)

        for i in range(self.n_epochs):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(indices, size=self.b_size, replace=False)

                self.X_part = self.X_full[chosen_indices]
                self.Y_part = self.Y_full[chosen_indices]

                self.feed_forward()
                self.backpropagation()

"""

BACKUP

def get_B_W(self):
    self.hid_W_first = np.random.randn(self.n_features, self.n_hid_neur)
    self.hid_B_first = np.zeros(self.n_hid_neur) + 0.01

    self.hid_W_inner = np.zeros((self.hid_layers -1, self.n_hid_neur, self.n_hid_neur))
    for i in range(self.hid_layers - 1):
        self.hid_W_inner[i] = np.random.randn(self.n_hid_neur, self.n_hid_neur)
    self.hid_B_inner = np.zeros((self.hid_layers - 1, self.n_hid_neur)) + 0.01

    self.out_W = np.random.randn(self.n_hid_neur, self.n_cat)
    self.out_B = np.zeros(self.n_cat) + 0.01

class NeuralLogReg(NeuralNetwork):

    def feed_forward(self):
        self.zh = self.X_part @ self.hid_W + self.hid_B
        self.ah = self.sigmoid(self.zh)
        self.zo = self.ah @ self.out_W + self.out_B
        self.probs = self.sigmoid(self.zo)

    def feed_forward_out(self, X):
        zh = X @ self.hid_W + self.hid_B
        ah = self.sigmoid(zh)
        zo = ah @ self.out_W + self.out_B
        probs = self.sigmoid(zo)

        return probs

    def backpropagation(self):
        error_o = self.probs - self.Y_part
        error_h = np.multiply(np.multiply((error_o @ self.out_W.T), self.ah), (1 - self.ah))

        self.out_W_grad = self.ah.T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)

        self.hid_W_grad = self.X_part.T @ error_h
        self.hid_B_grad = np.sum(np.asarray(error_h), axis=0)

        if self.lmbd > 0.0:
            self.out_W_grad += self.lmbd*self.out_W
            self.hid_W_grad += self.lmbd*self.hid_W

        self.out_W -= self.eta*self.out_W_grad
        self.out_B -= self.eta*self.out_B_grad
        self.hid_W -= self.eta*self.hid_W_grad
        self.hid_B -= self.eta*self.hid_B_grad

    def predict(self, X):
        probs = self.feed_forward_out(X)
        return np.argmax(probs, axis=1)

    def predict_probabilities(self, X):
        probs = self.feed_forward_out(X)
        return probs

    def train(self):
        indices = np.arange(self.n_inputs)

        for i in tqdm.tqdm(range(self.n_epochs)):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(indices, size=self.b_size, replace=False)

                self.X_part = self.X_full[chosen_indices]
                self.Y_part = self.Y_full[chosen_indices]

                self.feed_forward()
                self.backpropagation()
class NeuralLinReg(NeuralNetwork):

    def feed_forward(self):
        self.zh = self.X_part @ self.hid_W + self.hid_B
        self.ah = self.sigmoid(self.zh)
        self.zo = self.ah @ self.out_W + self.out_B
        self.zn = self.sigmoid(self.zo)


    def feed_forward_out(self, X):
        zh = X @ self.hid_W + self.hid_B
        ah = self.sigmoid(zh)
        zo = ah @ self.out_W + self.out_B

        return self.sigmoid(zo)


    def backpropagation(self):
        error_o = 2*(self.zn - self.Y_part)/len(self.zo)
        #error_h = error_o @ self.out_W.T
        error_h = np.multiply(np.multiply((error_o @ self.out_W.T), self.ah), (1 - self.ah))

        self.out_W_grad = self.ah.T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)



        self.hid_W_grad = self.X_part.T @ error_h
        self.hid_B_grad = np.sum(np.asarray(error_h), axis=0)

        if self.lmbd > 0.0:
            self.out_W_grad += self.lmbd*self.out_W
            self.hid_W_grad += self.lmbd*self.hid_W

        self.out_W -= self.eta*self.out_W_grad
        self.out_B -= self.eta*self.out_B_grad
        self.hid_W -= self.eta*self.hid_W_grad
        self.hid_B -= self.eta*self.hid_B_grad

    def predict(self, X):
        z = self.feed_forward_out(X)
        return z


    def train(self):
        indices = np.arange(self.n_inputs)

        for i in tqdm.tqdm(range(self.n_epochs)):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(indices, size=self.b_size, replace=False)

                self.X_part = self.X_full[chosen_indices]
                self.Y_part = self.Y_full[chosen_indices]

                self.feed_forward()
                self.backpropagation()
"""
