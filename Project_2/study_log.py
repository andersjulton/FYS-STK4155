import numpy as np
import tqdm
from readFile import *
import matplotlib.pyplot as plt
import scikitplot as skplt
from logClass import GradientDescent, StochasticGradient, NewtonRaphsons
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

onehotencoder = OneHotEncoder(categories="auto")


comp_GD = False
comp_SGD = True
comp_NR = False


X, y = readfile()
y = y.reshape(y.size)

trainingShare = 0.5
seed  = 2
XTrain, XTest, yTrain, yTest = train_test_split(X, y,
    train_size=trainingShare,
    random_state=seed)

sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

"""
Compare with sklearn

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(XTrain, yTrain)
preds = clf.predict(XTest)
"""

if comp_GD:
    gd = GradientDescent()
    gd.fit(XTrain, yTrain)

    yPredTest = gd(XTest)
    yPredTrain = gd(XTrain)

    scoreTest = gd.accuracy(yTest, yPredTest)
    scoreTrain = gd.accuracy(yTrain, yPredTrain)
    trainAreaRatio = gd.get_Area_ratio(yTrain, yPredTrain)
    testAreaRatio = gd.get_Area_ratio(yTest, yPredTest)

    #gd.plot(yTest, yPredTest)
    #gd.plot(yTrain, yPredTrain)
    print("Scores for GD method")
    print("Test score = ", scoreTest)
    print("Train score = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_SGD:
    sgd = StochasticGradient(max_iter = 5000, n_epochs = 80)
    sgd.fit(XTrain, yTrain)

    yPredTest = sgd(XTest)
    yPredTrain = sgd(XTrain)

    scoreTest = sgd.accuracy(yTest, yPredTest)
    scoreTrain = sgd.accuracy(yTrain, yPredTrain)

    trainAreaRatio = sgd.get_Area_ratio(yTrain, yPredTrain)
    testAreaRatio = sgd.get_Area_ratio(yTest, yPredTest)

    sgd.plot(yTest, yPredTest)
    sgd.plot(yTrain, yPredTrain)
    print("Scores for SGD method")
    print("Test score = ", scoreTest)
    print("Train score = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_NR:

    """
    Struggles with large data set. Memory error. Singluar matrix.
    Might be linalg solutions. SVD too slow.
    """

    nr = NewtonRaphsons()
    nr.fit(XTrain, yTrain)

    yPredTest = nr(XTest)

    scoreTest = nr.accuracy(yTest, yPredTest)

    print(scoreTest)
