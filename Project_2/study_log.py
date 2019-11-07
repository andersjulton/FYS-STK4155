import numpy as np
from readFile import *
from logClass import GradientDescent, StochasticGradient, StochasticGradientMiniBatch, ADAM
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

comp_ADAM = False
comp_GD = False
comp_SGD = True
comp_SGD_MB = False
comp_sklearn = False
comp_GD_resampling = False
comp_GD_change_remove = False

figpath = "figures/"
respath = "results/"
datapath = "datafiles/"

def get_credit_data(up_sample=False, down_sample=False):

    onehotencoder = OneHotEncoder(categories="auto")

    X, y = readfile()

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
    XTrain_sub = sc.fit_transform(XTrain[:,32:-1])
    XTest_sub = sc.transform(XTest[:,32:-1])

    XTrain = np.concatenate((XTrain[:,0:31], XTrain_sub), axis=1)
    XTest = np.concatenate((XTest[:,0:31], XTest_sub), axis=1)

    test = [XTest, yTest]
    train = [XTrain, yTrain]

    return test, train

if comp_ADAM:

    test, train = get_credit_data(down_sample=True)

    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)

    gd = ADAM(b_size = 256, n_epochs = 1000000//256)
    gd.fit(XTrain, yTrain)

    yPredTest = gd(XTest)
    yPredTrain = gd(XTrain)

    scoreTest = gd.accuracy(yTest, yPredTest)
    scoreTrain = gd.accuracy(yTrain, yPredTrain)

    trainAreaRatio = gd.get_Area_ratio(yTrain, yPredTrain)
    testAreaRatio = gd.get_Area_ratio(yTest, yPredTest)

    gd.plot(yTest, yPredTest, figpath + "ADAM_credit")

    file = open(respath + "ADAM_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % roc_auc_score(yTest, yPredTest))
    file.write("Sklearn F1 = %1.4f\n" % f1_score(yTest, np.round(yPredTest)))
    file.write("Test accuracy = %1.4f\n" % scoreTest)
    file.write("Train accuracy = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for ADAM method")
    print("Sklearn AUC Score: ", roc_auc_score(yTest, yPredTest))
    print("Sklearn F1: ", f1_score(yTest, np.round(yPredTest)))
    print("Test accuracy = ", scoreTest)
    print("Train accuracy = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)


if comp_GD:

    test, train = get_credit_data(down_sample=True)

    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)

    gd = GradientDescent(eta = 0.001)
    gd.fit(XTrain, yTrain)

    yPredTest = gd(XTest)
    yPredTrain = gd(XTrain)

    scoreTest = gd.accuracy(yTest, yPredTest)
    scoreTrain = gd.accuracy(yTrain, yPredTrain)

    trainAreaRatio = gd.get_Area_ratio(yTrain, yPredTrain)
    testAreaRatio = gd.get_Area_ratio(yTest, yPredTest)

    gd.plot(yTest, yPredTest, figpath + "GD_credit")

    file = open(respath + "GD_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % roc_auc_score(yTest, yPredTest))
    file.write("Sklearn F1 = %1.4f\n" % f1_score(yTest, np.round(yPredTest)))
    file.write("Test accuracy = %1.4f\n" % scoreTest)
    file.write("Train accuracy = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for GD method")
    print("Sklearn AUC Score: ", roc_auc_score(yTest, yPredTest))
    print("Sklearn F1: ", f1_score(yTest, np.round(yPredTest)))
    print("Test accuracy = ", scoreTest)
    print("Train accuracy = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_SGD:
    test, train = get_credit_data(down_sample=True)

    XTest, yTest = test
    XTrain, yTrain = train

    yTrain = np.ravel(yTrain)

    sgd = StochasticGradient(n_epochs = 100, eta=0.001, gamma=0.9)
    sgd.fit(XTrain, yTrain)

    yPredTest = sgd(XTest)
    yPredTrain = sgd(XTrain)

    scoreTest = sgd.accuracy(yTest, yPredTest)
    scoreTrain = sgd.accuracy(yTrain, yPredTrain)

    trainAreaRatio = sgd.get_Area_ratio(yTrain, yPredTrain)
    testAreaRatio = sgd.get_Area_ratio(yTest, yPredTest)

    sgd.plot(yTest, yPredTest, figpath + "SGD_credit")

    file = open(respath + "SGD_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % roc_auc_score(yTest, yPredTest))
    file.write("Sklearn F1 = %1.4f\n" % f1_score(yTest, np.round(yPredTest)))
    file.write("Test accuracy = %1.4f\n" % scoreTest)
    file.write("Train accuracy = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for SGD method")
    print("Sklearn AUC Score: ", roc_auc_score(yTest, yPredTest))
    print("Sklearn F1: ", f1_score(yTest, np.round(yPredTest)))
    print("Test accuracy = ", scoreTest)
    print("Train accuracy = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_SGD_MB:
    test, train = get_credit_data(down_sample=True)
    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)

    sgd_mb = StochasticGradientMiniBatch(b_size = 256, n_epochs = 1000, eta = 0.001)
    sgd_mb.fit(XTrain, yTrain)

    yPredTest = sgd_mb(XTest)
    yPredTrain = sgd_mb(XTrain)

    scoreTest = sgd_mb.accuracy(yTest, yPredTest)
    scoreTrain = sgd_mb.accuracy(yTrain, yPredTrain)

    testAreaRatio = sgd_mb.get_Area_ratio(yTest, yPredTest)
    trainAreaRatio = sgd_mb.get_Area_ratio(yTrain, yPredTrain)

    sgd_mb.plot(yTest, yPredTest, figpath + "SGD_MB_credit")

    file = open(respath + "SGD_MB_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % roc_auc_score(yTest, yPredTest))
    file.write("Sklearn F1 = %1.4f\n" % f1_score(yTest, np.round(yPredTest)))
    file.write("Test accuracy = %1.4f\n" % scoreTest)
    file.write("Train accuracy = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for SGD_MB method")
    print("Sklearn AUC Score: ", roc_auc_score(yTest, yPredTest))
    print("Sklearn F1: ", f1_score(yTest, np.round(yPredTest)))
    print("Test accuracy = ", scoreTest)
    print("Train accuracy = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_sklearn:

    gd = GradientDescent()

    test, train = get_credit_data(down_sample=True)
    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)
    yTest = np.ravel(yTest)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto').fit(XTrain, yTrain)

    yPredTest = clf.predict_proba(XTest)[:,1]
    yPredTrain = clf.predict_proba(XTrain)[:,1]

    scoreTest = clf.score(XTest, yTest)
    scoreTrain = clf.score(XTrain, yTrain)

    gd.plot(yTest.reshape(-1, 1), yPredTest, figpath + "sklearn_credit")
    testAreaRatio = gd.get_Area_ratio(yTest, yPredTest)
    trainAreaRatio = gd.get_Area_ratio(yTrain, yPredTrain)

    file = open(respath + "sklearn_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % roc_auc_score(yTest, yPredTest))
    file.write("Sklearn F1 = %1.4f\n" % f1_score(yTest, np.round(yPredTest)))
    file.write("Test accuracy = %1.4f\n" % scoreTest)
    file.write("Train accuracy = %1.4f\n" % scoreTrain)
    file.write("Test area ratio = %1.4f\n" % testAreaRatio)
    file.write("Train area ratio = %1.4f\n" % trainAreaRatio)
    file.close()

    print("Scores for SKlearn package")
    print("Sklearn AUC Score: ", roc_auc_score(yTest, yPredTest))
    print("Sklearn F1: ", f1_score(yTest, np.round(yPredTest)))
    print("Test accuracy = ", scoreTest)
    print("Train accuracy = ", scoreTrain)
    print("Test area ratio = ", testAreaRatio)
    print("Train area ratio = ", trainAreaRatio)

if comp_GD_change_remove:

    test, train = get_credit_data(change_values=True, remove_values=False, down_sample=False)

    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)

    eta = np.linspace(-6, -1, 6)
    data = np.zeros(len(eta), dtype=object)
    areaRatio = np.zeros((6, len(eta)))

    run = False

    if run:
        for i, e in enumerate(eta):
            gd = GradientDescent(max_iter = 10000, eta = 10**e)
            gd.fit(XTrain, yTrain)
            data[i] = gd

        np.save(datapath + "GD_credit_change", data)

    else:
        datas = np.load(datapath + "GD_credit_standard.npy", allow_pickle=True)
        datar = np.load(datapath + "GD_credit_remove.npy", allow_pickle=True)
        datac = np.load(datapath + "GD_credit_change.npy", allow_pickle=True)

    for i in range(len(eta)):
        gds = datas[i]
        gdr = datar[i]
        gdc = datac[i]

        yPredTests = gds(XTest)
        yPredTrains = gds(XTrain)
        yPredTestr = gdr(XTest)
        yPredTrainr = gdr(XTrain)
        yPredTestc = gdc(XTest)
        yPredTrainc = gdc(XTrain)

        areaRatio[0][i] = gds.get_Area_ratio(yTest, yPredTests)
        areaRatio[1][i] = gds.get_Area_ratio(yTrain, yPredTrains)
        areaRatio[2][i] = gdr.get_Area_ratio(yTest, yPredTestr)
        areaRatio[3][i] = gdr.get_Area_ratio(yTrain, yPredTrainr)
        areaRatio[4][i] = gdc.get_Area_ratio(yTest, yPredTestc)
        areaRatio[5][i] = gdc.get_Area_ratio(yTrain, yPredTrainc)

    plt.plot(eta, areaRatio[0], label="Ratio test standard")
    plt.plot(eta, areaRatio[1], label="Ratio train standard")
    #plt.plot(eta, areaRatio[2], label="Ratio test remove")
    #plt.plot(eta, areaRatio[3], label="Ratio train remove")
    plt.plot(eta, areaRatio[4], label="Ratio test change")
    plt.plot(eta, areaRatio[5], label="Ratio train change")
    plt.legend()
    #plt.savefig(figpath + "GD_credit_eta_standard.pdf")
    plt.show()

if comp_GD_resampling:

    test, train = get_credit_data(change_values=False, remove_values=False, up_sample=True)

    XTest, yTest = test
    XTrain, yTrain = train
    yTrain = np.ravel(yTrain)

    eta = np.linspace(-6, -1, 6)
    data = np.zeros(len(eta), dtype=object)
    areaRatio = np.zeros((6, len(eta)))

    run = False

    if run:
        for i, e in enumerate(eta):
            gd = GradientDescent(max_iter = 10000, eta = 10**e)
            gd.fit(XTrain, yTrain)
            data[i] = gd

        np.save(datapath + "GD_credit_up", data)

    else:
        datad = np.load(datapath + "GD_credit_down.npy", allow_pickle=True)
        datau = np.load(datapath + "GD_credit_up.npy", allow_pickle=True)
        data = np.load(datapath + "GD_credit_standard.npy", allow_pickle=True)


    for i in range(len(eta)):
        gd = data[i]
        gdd = datad[i]
        gdu = datau[i]

        yPredTest = gd(XTest)
        yPredTrain = gd(XTrain)
        yPredTestd = gdd(XTest)
        yPredTraind = gdd(XTrain)
        yPredTestu = gdu(XTest)
        yPredTrainu = gdu(XTrain)

        areaRatio[0][i] = gd.get_Area_ratio(yTest, yPredTest)
        areaRatio[1][i] = gd.get_Area_ratio(yTrain, yPredTrain)
        areaRatio[2][i] = gdd.get_Area_ratio(yTest, yPredTestd)
        areaRatio[3][i] = gdd.get_Area_ratio(yTrain, yPredTraind)
        areaRatio[4][i] = gdu.get_Area_ratio(yTest, yPredTestu)
        areaRatio[5][i] = gdu.get_Area_ratio(yTrain, yPredTrainu)

    plt.plot(eta, areaRatio[0], label="Ratio test ")
    plt.plot(eta, areaRatio[1], label="Ratio train ")
    plt.plot(eta, areaRatio[2], label="Ratio test RandomDownSampler")
    plt.plot(eta, areaRatio[3], label="Ratio train RandomDownSampler")
    plt.plot(eta, areaRatio[4], label="Ratio test SMOTE")
    plt.plot(eta, areaRatio[5], label="Ratio train SMOTE")
    plt.legend()
    plt.xlabel(r"$\eta$")
    plt.savefig(figpath + "GD_credit_eta_sdu_compare.pdf")
    plt.show()


    """
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
    print("Train area ratio = ", trainAreaRatio)"""
