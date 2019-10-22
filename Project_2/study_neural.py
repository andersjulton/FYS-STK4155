import numpy as np
from readFile import *
from neuralClass import NeuralLinReg, NeuralLogReg, NeuralNetwork
from franke import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

comp_credit = False
comp_sklearn = False
comp_franke = True

def get_credit_data():

    X, y = readfile(True)

    trainingShare = 0.5
    seed  = 2
    XTrain, XTest, yTrain, yTest = train_test_split(X, y,
        train_size=trainingShare,
        random_state=seed)

    onehotencoder = OneHotEncoder(categories="auto")

    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    yTrain_onehot, yTest_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)

    test = [XTest, yTest, yTest_onehot]
    train = [XTrain, yTrain, yTrain_onehot]
    return test, train

if comp_credit:
    test, train = get_credit_data()
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    neur = NeuralNetwork(XTrain, yTrain_onehot, n_hid_neur=50, n_cat=2, n_epochs=100, b_size=80, eta=0.0001, lmbd=0.1, hid_layers=4)
    neur.train()
    yPredTest = neur.predict(XTest)

    print((yPredTest == yTest).mean())

if comp_sklearn:
    test, train = get_credit_data()
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    n_hid_neur = 50; n_epochs=100; eta=0.1; lmbd=0.1

    dnn = MLPClassifier(hidden_layer_sizes=n_hid_neur, activation='logistic',
    alpha=lmbd, learning_rate_init=eta, max_iter=n_epochs)
    dnn.fit(XTrain, yTrain_onehot)

    print(dnn.score(XTest, yTest_onehot))

if comp_franke:
    #np.random.seed(42)
    def R2(z, z_tilde):
        RR_res = np.sum((z - z_tilde)**2)
        RR_tot = np.sum((z - np.mean(z))**2)
        return 1 - RR_res/RR_tot

    def MSE(z, z_tilde):
        return np.mean((z - z_tilde)**2)


    n = 100; p = 5
    x, y, z = get_train_data(n, noise=False)

    X = CreateDesignMatrix(x, y, p)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y,
        train_size=0.6,
        random_state=0)
    yTrain = yTrain.reshape(-1,1)


    neurLin = NeuralLinReg(XTrain, yTrain, n_cat = 1, eta = 0.001, lmbd=0.5, hid_layers = 3)
    neurLin.train()

    z = neurLin.feed_forward_out(XTest)

    print(MSE(yTest, np.ravel(z)))
    print(R2(yTest, np.ravel(z)))
