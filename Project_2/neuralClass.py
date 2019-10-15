import numpy as np
import tqdm


class NeuralNetwork:
    def __init__(self, X, y, n_hid_neur=50, n_cat=2, n_epochs=100, b_size=100, eta=0.1, lmbd=0.1):

        self.X_full = X
        self.Y_full = y

        self.n_inputs = X.shape[0]
        self.n_features = X.shape[1]
        self.n_hid_neur = n_hid_neur
        self.n_cat = n_cat

        self.n_epochs = n_epochs
        self.b_size = b_size
        self.iterations = self.n_inputs // self.b_size
        self.eta = eta
        self.lmbd = lmbd

        self.get_B_W()

    def get_B_W(self):
        self.hid_W = np.random.randn(self.n_features, self.n_hid_neur)
        self.hid_B = np.zeros(self.n_hid_neur) + 0.01

        self.out_W = np.random.randn(self.n_hid_neur, self.n_cat)
        self.out_B = np.zeros(self.n_cat) + 0.01

    def feed_forward(self):
        # feed-forward for training
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

        for i in tqdm.tqdm(range(self.n_epochs)):
            for j in range(self.iterations):
                chosen_indices = np.random.choice(indices, size=self.b_size, replace=False)

                self.X_part = self.X_full[chosen_indices]
                self.Y_part = self.Y_full[chosen_indices]

                self.feed_forward()
                self.backpropagation()

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


    def feed_forward_out(self, X):
        zh = X @ self.hid_W + self.hid_B
        ah = self.sigmoid(zh)
        zo = ah @ self.out_W + self.out_B

        return zo

        """
        cost function: np.mean((a - y)^2)
        """
    def backpropagation(self):
        error_o = 2*(self.zo - self.Y_part)/len(self.zo)
        error_h = self.out_W.T @ error_o
        self.out_W_grad = self.ah.T @ error_o
        self.out_B_grad = np.sum(np.asarray(error_o), axis=0)
        print(self.out_B_grad.shape)
        print(self.out_W_grad.shape)
        print(self.out_W.shape)
        print(self.ah.shape)
        print(error_o.shape)
        input()

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
