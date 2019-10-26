import numpy as np
from readFile import *
from logClass import GradientDescent, StochasticGradient

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

comp_GD = False
comp_SGD = False
comp_sklearn = False

figpath = "figures/"
respath = "results/"

def get_credit_data(change_values=True, remove_values=False, up_sample=False, down_sample=False):


    onehotencoder = OneHotEncoder(categories="auto")

    X, y = readfile(change_values, remove_values)

    trainingShare = 0.5
    seed  = 2
    XTrain, XTest, yTrain, yTest = train_test_split(X, y,
        train_size=trainingShare,
        random_state=seed)

    if up_sample:
        XTrain, yTrain = SMOTE().fit_resample(XTrain, np.ravel(yTrain))
    elif down_sample:
        XTrain, yTrain = RandomUnderSampler(random_state=1).fit_resample(XTrain, yTrain)

    #yTrain_onehot, yTest_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)

    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    test = [XTest, yTest]
    train = [XTrain, yTrain]

    return test, train


if comp_GD:

    test, train = get_credit_data(change_values=True, remove_values=False, down_sample=True)

    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)

    gd = GradientDescent()
    gd.fit(XTrain, yTrain)

    yPredTest = gd(XTest)
    yPredTrain = gd(XTrain)

    scoreTest = gd.accuracy(yTest, yPredTest)
    scoreTrain = gd.accuracy(yTrain, yPredTrain)

    trainAreaRatio = gd.get_Area_ratio(yTrain, yPredTrain)
    testAreaRatio = gd.get_Area_ratio(yTest, yPredTest)

    gd.plot(yTest, yPredTest, figpath + "GD_credit_test_down")
    gd.plot(yTrain, yPredTrain, figpath + "GD_credit_train_down")

    file = open(respath + "GD_results_down.txt", "w+")

    file.write("Test score = %1.4f\n" % scoreTest)
    file.write("Train score = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for GD method")
    print("Test score = ", scoreTest)
    print("Train score = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_SGD:
    test, train = get_credit_data(change_values=True, remove_values=False, down_sample=False)

    XTest, yTest = test
    XTrain, yTrain = train

    yTrain = np.ravel(yTrain)

    sgd = StochasticGradient(max_iter = 5000, n_epochs = 100)
    sgd.fit(XTrain, yTrain)

    yPredTest = sgd(XTest)
    yPredTrain = sgd(XTrain)

    scoreTest = sgd.accuracy(yTest, yPredTest)
    scoreTrain = sgd.accuracy(yTrain, yPredTrain)

    trainAreaRatio = sgd.get_Area_ratio(yTrain, yPredTrain)
    testAreaRatio = sgd.get_Area_ratio(yTest, yPredTest)

    sgd.plot(yTest, yPredTest, figpath + "SGD_credit_test")
    sgd.plot(yTrain, yPredTrain, figpath + "SGD_credit_train")

    file = open(respath + "SGD_results.txt", "w+")

    file.write("Test score = %1.4f\n" % scoreTest)
    file.write("Train score = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for SGD method")
    print("Test score = ", scoreTest)
    print("Train score = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_sklearn:

    gd = GradientDescent()

    test, train = get_credit_data(change_values=True, remove_values=False, down_sample=True)
    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)
    yTest = np.ravel(yTest)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto').fit(XTrain, yTrain)

    yPredTest = clf.predict_proba(XTest)[:,1]
    yPredTrain = clf.predict_proba(XTrain)[:,1]

    scoreTest = clf.score(XTest, yTest)
    scoreTrain = clf.score(XTrain, yTrain)

    gd.plot(yTest.reshape(-1, 1), yPredTest, figpath + "sklearn_credit_test_down")
    gd.plot(yTrain.reshape(-1, 1), yPredTrain, figpath + "sklearn_credit_train_down")
    testAreaRatio = gd.get_Area_ratio(yTest, yPredTest)
    trainAreaRatio = gd.get_Area_ratio(yTrain, yPredTrain)

    file = open(respath + "sklearn_results_down.txt", "w+")

    file.write("Test score = %1.4f\n" % scoreTest)
    file.write("Train score = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for SKlearn")
    print("Test score = ", scoreTest)
    print("Train score = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)
