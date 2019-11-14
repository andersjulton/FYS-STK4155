import numpy as np
from readFile import *
from logClass import GradientDescent, StochasticGradient, StochasticGradientMiniBatch, ADAM
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

comp_ADAM = False
comp_GD = False
comp_SGD = False
comp_SGD_MB = False
comp_sklearn = False
comp_resampling = False
comp_down = False

figpath = "figures/"
respath = "results/"
datapath = "datafiles/"

def get_credit_data(up_sample=False, down_sample=False, d_ratio = 1):

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
        XTrain, yTrain = RandomUnderSampler(random_state=1, sampling_strategy = d_ratio).fit_resample(XTrain, yTrain)

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

    gd = ADAM(b_size = 1000, n_epochs = 200)
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
    print("F1 test: ", f1_score(yTest, np.round(yPredTest)))
    print("F1 train: ", f1_score(yTrain, np.round(yPredTrain)))
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

    sgd_mb = StochasticGradientMiniBatch(b_size = 500, n_epochs = 100, eta = 0.001)
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

if comp_down:

    ratio = np.linspace(0.3, 1, 16)
    results = np.zeros(len(ratio))

    run = True

    if run:
        for i, r in enumerate(tqdm(ratio)):
            result = 0
            for j in range(5):
                test, train = get_credit_data(down_sample=True, d_ratio=r)
                XTest, yTest = test
                XTrain, yTrain = train
                yTrain = np.ravel(yTrain)
                adam = ADAM(n_epochs = 1000, eta = 1e-2)
                adam.fit(XTrain, yTrain)
                yPredTest = adam(XTest)
                result += adam.get_Area_ratio(yTest, yPredTest)
            results[i] = result/5

        np.save(datapath + "down_sample_test", results)
    else:
        results = np.load(datapath + "down_sample_test.npy")

    plt.plot(ratio, results)
    plt.xlabel(r"$N_1/N_0$", fontsize=12)
    plt.ylabel("Area ratio", fontsize=12)
    plt.ylim((np.min(results) - 0.005, np.max(results) + 0.005))
    plt.tight_layout()
    plt.savefig(figpath + "down_ratio_compare.pdf")
    plt.show()

if comp_resampling:

    testu, trainu = get_credit_data(up_sample=True)
    testd, traind = get_credit_data(down_sample=True)
    test, train = get_credit_data(down_sample=False, up_sample=False)

    XTestu, yTestu = testu
    XTestd, yTestd = testd
    XTest, yTest = test
    XTrainu, yTrainu = trainu
    XTraind, yTraind = traind
    XTrain, yTrain = train
    yTrainu = np.ravel(yTrainu)
    yTraind = np.ravel(yTraind)
    yTrain = np.ravel(yTrain)

    eta = np.logspace(-6, -1, 6)
    data = np.zeros(len(eta), dtype=object)
    datau = data.copy()
    datad = data.copy()
    areaRatio = np.zeros((3, len(eta)))
    F1score = areaRatio.copy()

    run = False

    if run:
        for i, e in enumerate(tqdm(eta)):
            adam = ADAM( n_epochs = 1000, eta = e)
            adamu = ADAM(n_epochs = 1000, eta = e)
            adamd = ADAM(n_epochs = 1000, eta = e)
            adam.fit(XTrain, yTrain)
            adamu.fit(XTrainu, yTrainu)
            adamd.fit(XTraind, yTraind)
            data[i] = adam
            datau[i] = adamu
            datad[i] = adamd

        np.save(datapath + "adam_credit_standard", data)
        np.save(datapath + "adam_credit_up", datau)
        np.save(datapath + "adam_credit_down", datad)


    else:
        datad = np.load(datapath + "adam_credit_down.npy", allow_pickle=True)
        datau = np.load(datapath + "adam_credit_up.npy", allow_pickle=True)
        data = np.load(datapath + "adam_credit_standard.npy", allow_pickle=True)


    for i in range(len(eta)):
        gd = data[i]
        gdd = datad[i]
        gdu = datau[i]

        yPredTest = gd(XTest)
        yPredTestd = gdd(XTest)
        yPredTestu = gdu(XTest)

        areaRatio[0][i] = gd.get_Area_ratio(yTest, yPredTest)
        areaRatio[1][i] = gdd.get_Area_ratio(yTest, yPredTestd)
        areaRatio[2][i] = gdu.get_Area_ratio(yTest, yPredTestu)
        F1score[0][i] = f1_score(yTest, np.round(yPredTest))
        F1score[1][i] = f1_score(yTest, np.round(yPredTestd))
        F1score[2][i] = f1_score(yTest, np.round(yPredTestu))

    #plt.semilogx(eta, areaRatio[0], label="No resampling")
    #plt.semilogx(eta, areaRatio[1], label="RandomDownSampler")
    #plt.semilogx(eta, areaRatio[2], label="SMOTE")
    plt.semilogx(eta, F1score[0], label="No resampling")
    plt.semilogx(eta, F1score[1], label="RandomDownSampler")
    plt.semilogx(eta, F1score[2], label="SMOTE")
    plt.legend(fontsize=12)
    plt.ylabel("F1 score", fontsize=12)
    plt.xlabel(r"$\eta$",fontsize=12)
    plt.savefig(figpath + "adam_credit_resampling_compare_F1.pdf")
    plt.show()
