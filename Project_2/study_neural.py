import numpy as np
from readFile import *
from neuralClass import NeuralNetwork

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X, y = readfile(True)

trainingShare = 0.5
seed  = 1
XTrain, XTest, yTrain, yTest = train_test_split(X, y,
    train_size=trainingShare,
    random_state=seed)

onehotencoder = OneHotEncoder(categories="auto")

sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

yTrain_onehot, yTest_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)


neur = NeuralNetwork(XTrain, yTrain_onehot)

neur.train()
yPredTest = neur.predict(XTest)

print((yPredTest == yTest).mean())
