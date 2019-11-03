import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt


class NeuralNetwork(object):

    def __init__(self, X, y, hid_actFunc, out_actFunc,
        n_epochs=100,
        b_size=100,
        eta=0.1,
        lmbd=0.1,
        hid_layers=1):

        self.X_full = X
        self.Y_full = y
        self.hid_actFunc = hid_actFunc
        self.out_actFunc = out_actFunc

        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_cat = y.shape[1]
        self.n_hid_neur = int(np.mean(self.n_features + self.n_cat))
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
            self.hid_W_inner = np.zeros((self.hid_layers - 1, self.n_hid_neur, self.n_hid_neur))
            for i in range(self.hid_layers - 1):
                self.hid_W_inner[i] = np.random.randn(self.n_hid_neur, self.n_hid_neur)
            self.hid_B_inner = np.zeros((self.hid_layers - 1, self.n_hid_neur)) + 0.01

        self.out_W = np.random.randn(self.n_hid_neur, self.n_cat)
        self.out_B = np.zeros(self.n_cat) + 0.01


    def feed_forward(self):
        pass

    def feed_forward_out(self, X):
        pass

    def backpropagation(self):
        pass

    def predict(self, X):
        pass

    def predict_probabilities(self, X):
        pass

    def train(self):
        pass


class NeuralLogReg(NeuralNetwork):

    def cost(self, z, ztilde):
        return -np.mean(z.T @ np.log(self.sigmoid(ztilde)) + (np.ones(z.shape) - z).T @ np.log(self.sigmoid(-ztilde)))


    def feed_forward(self):
        self.zh = self.X_part @ self.hid_W_first + self.hid_B_first
        self.ah = np.zeros((self.hid_layers, self.b_size, self.n_hid_neur))
        self.ahderiv = np.zeros((self.hid_layers, self.b_size, self.n_hid_neur))
        self.ah[0] = self.hid_actFunc(self.zh)
        self.ahderiv[0] = self.hid_actFunc(self.zh, deriv=True)
        zh_next = 0

        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = self.ah[i] @ self.hid_W_inner[i] + self.hid_B_inner[i]
                self.ah[i+1] = self.hid_actFunc(zh_next)
                self.ahderiv[i+1] = self.hid_actFunc(zh_next, deriv=True)

        self.zo = self.ah[-1] @ self.out_W + self.out_B
        self.probs = self.out_actFunc(self.zo)


    def feed_forward_out(self, X):
        zh = X @ self.hid_W_first + self.hid_B_first
        ah = self.hid_actFunc(zh)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = ah @ self.hid_W_inner[i] + self.hid_B_inner[i]
                ah = self.hid_actFunc(zh_next)
        zo = ah @ self.out_W + self.out_B
        probs = self.out_actFunc(zo)
        return probs


    def backpropagation(self):
        error_o = self.probs - self.Y_part
        error_h_prev = np.multiply((error_o @ self.out_W.T), self.ahderiv[-1])
        self.out_W_grad = self.ah[-1].T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 2, 0, -1):
                error_h_next = np.multiply((error_h_prev @ self.hid_W_inner[i].T), self.ahderiv[i])
                hid_W_grad = self.ah[i].T @ error_h_next
                hid_B_grad = np.sum(np.asarray(error_h_next), axis=0)

                hid_W_grad += self.lmbd*self.hid_W_inner[i]
                self.hid_W_inner[i] -= self.eta*hid_W_grad
                self.hid_B_inner[i] -= self.eta*hid_B_grad
                error_h_prev = error_h_next

        hid_W_grad = self.X_part.T @ error_h_prev
        hid_B_grad = np.sum(np.asarray(error_h_prev), axis=0)


        self.out_W_grad += self.lmbd*self.out_W
        hid_W_grad += self.lmbd*self.hid_W_first

        self.out_W -= self.eta*self.out_W_grad
        self.out_B -= self.eta*self.out_B_grad
        self.hid_W_first -= self.eta*hid_W_grad
        self.hid_B_first -= self.eta*hid_B_grad


    def predict(self, X):
        probs = self.feed_forward_out(X)
        return np.argmax(probs, axis=1)


    def predict_proba(self, X):
        probs = self.feed_forward_out(X)
        return probs


    def train(self):
        indices = np.arange(self.n_inputs)

        for i in range(self.n_epochs):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(indices, size=self.b_size, replace=False)

                self.X_part = self.X_full[chosen_indices]
                self.Y_part = self.Y_full[chosen_indices]

                self.feed_forward()
                self.backpropagation()
            #print("Cost score = %1.4f" % self.cost(self.Y_part, self.probs), end="\r")


    def get_Area_ratio(self, y, ypred):
        ax = skplt.metrics.plot_cumulative_gain(y, ypred)
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


    def sigmoid(self, z, deriv=False):
        sigm = 1/(1 + np.exp(-z))
        if deriv:
            return sigm*(1 - sigm)
        else:
            return sigm


class NeuralLinReg(NeuralNetwork):

    def cost(self, z, ztilde):
        return np.mean((z - ztilde)**2)


    def feed_forward(self):
        self.zh = self.X_part @ self.hid_W_first + self.hid_B_first
        self.ah = np.zeros((self.hid_layers, self.b_size, self.n_hid_neur))
        self.ahderiv = self.ah.copy()
        self.ah[0] = self.hid_actFunc(self.zh)
        self.ahderiv[0] = self.hid_actFunc(self.zh, deriv=True)
        zh_next = 0

        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = self.ah[i] @ self.hid_W_inner[i] + self.hid_B_inner[i]
                self.ah[i+1] = self.hid_actFunc(zh_next)
        self.zo = self.ah[-1] @ self.out_W + self.out_B
        self.ao = self.out_actFunc(self.zo)


    def feed_forward_out(self, X):
        zh = X @ self.hid_W_first + self.hid_B_first
        ah = self.hid_actFunc(zh)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 1):
                zh_next = ah @ self.hid_W_inner[i] + self.hid_B_inner[i]
                ah = self.hid_actFunc(zh_next)
        zo = ah @ self.out_W + self.out_B
        ao = self.out_actFunc(zo)
        return ao


    def backpropagation(self):
        error_o = 2*(self.ao - self.Y_part)/len(self.ao)
        error_h_prev = np.multiply((error_o @ self.out_W.T), self.ahderiv[-1])

        self.out_W_grad = self.ah[-1].T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)
        if self.hid_layers > 1:
            for i in range(self.hid_layers - 2, 0, -1):
                error_h_next = np.multiply((error_h_prev @ self.hid_W_inner[i].T), self.ahderiv[i])
                hid_W_grad = self.ah[i].T @ error_h_next
                hid_B_grad = np.sum(np.asarray(error_h_next), axis=0)


                hid_W_grad += self.lmbd*self.hid_W_inner[i]
                self.hid_W_inner[i] -= self.eta*hid_W_grad
                self.hid_B_inner[i] -= self.eta*hid_B_grad
                error_h_prev = error_h_next

        hid_W_grad = self.X_part.T @ error_h_prev
        hid_B_grad = np.sum(np.asarray(error_h_prev), axis=0)


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


class ActivationFunctions(object):
    def __init__(self, a=0.01):
        self.a = a


    def sigmoid(self, z, deriv=False):
        sigm = 1/(1 + np.exp(-z))
        if deriv:
            return sigm*(1 - sigm)
        else:
            return sigm


    def ReLU(self, z, deriv=False):
        zn = np.zeros(z.shape)
        indices = np.where(z >= 0)
        if deriv:
            zn[indices] = 1
        else:
            zn[indices] = z[indices]
        return zn


    def PReLU(self, z, deriv=False):
        indices = np.where(z >= 0)
        if deriv:
            zn = np.zeros(z.shape) + self.a
            zn[indices] = 1
        else:
            zn = z*self.a
            zn[indices] = z[indices]
        return zn


    def tanh(self, z, deriv=False):
        if deriv:
            return 1 - np.tanh(z)**2
        else:
            return np.tanh(z)


    def identity(self, z, deriv=False):
        if deriv:
            return 1
        else:
            return z
