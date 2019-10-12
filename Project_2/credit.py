import numpy as np
import tqdm
from readFile import *
import matplotlib.pyplot as plt
import scikitplot as skplt
from logClass import GradientDescent, StochasticGradient, NewtonRaphsons

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

"""
KLADDEFIL!
"""

X, y = readfile()

trainingShare = 0.5
seed  = 1
XTrain, XTest, yTrain, yTest = train_test_split(X, y,
    train_size=trainingShare,
    random_state=seed)

# building our neural network

n_inputs, n_features = XTrain.shape
n_hidden_neurons = 50
n_categories = 2

# we make the weights normally distributed using numpy.random.randn

# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def feedForward(X):
    zh = X @ hidden_weights + hidden_bias
    ah = sigmoid(zh)
    zo = ah @ output_weights + output_bias

    return sigmoid(zo)

def backPropagation(X, y, prob):
    errorOut = (prob != y).mean()
    errorHid = errorOut @ output_weights.T @ ah @(1 - ah)

    outWgrad = ah.T @ errorOut
    outBgrad = np.sum(errorOut, axis=0)

    hidWgrad = X.T @ errorHid
    hidBgrad = np.sum(errorHid, axis=0)

    if lamb > 0:
        outWgrad += lmbd*output_weights
        hidWgrad += lmdb*hidden_weights

    output_weights -= eta

probabilities = feedForward(XTrain)

pred1 = probabilities[:,0].round()
pred2 = probabilities[:,1].round()


print((pred1 == yTrain).mean())
print((pred2 == yTrain).mean())
