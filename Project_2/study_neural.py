import numpy as np
from readFile import *
from neuralClass import NeuralLinReg, NeuralLogReg, ActivationFunctions
from franke import *
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings('ignore')

figpath = "figures/NN/"
datapath = "datafiles/"
respath = "results/NN/"


grid_search_credit = True
comp_sklearn = False
comp_franke = False

def get_credit_data(up_sample=False, down_sample=False):

    X, y = readfile()

    trainingShare = 0.6
    seed  = 3
    XTrain, XTest, yTrain, yTest = train_test_split(X, y,
        train_size=trainingShare,
        random_state=seed)

    onehotencoder = OneHotEncoder(categories="auto")

    sc = StandardScaler()
    XTrain = sc.fit_transform(XTrain)
    XTest = sc.transform(XTest)

    if up_sample:
        XTrain, yTrain = SMOTE().fit_resample(XTrain, np.ravel(yTrain))
    elif down_sample:
        XTrain, yTrain = RandomUnderSampler().fit_resample(XTrain, yTrain)

    yTrain = yTrain.reshape(-1, 1)
    yTrain_onehot, yTest_onehot = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)

    test = [XTest, yTest, yTest_onehot]
    train = [XTrain, yTrain, yTrain_onehot]
    return test, train

if grid_search_credit:
    test, train = get_credit_data(down_sample=True)
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    eta = np.logspace(-6, -1, 6)
    lmbd = np.logspace(-6, -1, 6)

    scoresArea = np.zeros((len(eta), len(lmbd)))
    scoresF1 = scoresArea.copy()
    scoresAUC = scoresArea.copy()

    NNdata = np.zeros((len(eta), len(lmbd)), dtype=object)

    hid_AF = ActivationFunctions().PReLU
    out_AF = ActivationFunctions().sigmoid

    run = False
    act = "_PReLU"

    if run:
        for i, e in enumerate(tqdm(eta)):
            for j, l in enumerate(lmbd):
                neur = NeuralLogReg(XTrain, yTrain_onehot,
                    hid_actFunc=hid_AF,
                    out_actFunc=out_AF,
                    n_epochs=100,
                    b_size=500,
                    eta=e,
                    lmbd=l,
                    hid_layers=1)
                neur.train()
                NNdata[i][j] = neur

        np.save(datapath + "NN_data_credit" + act, NNdata)

    else:
        NNdata = np.load(datapath + "NN_data_credit" + act + ".npy", allow_pickle=True)


    for i in range(len(eta)):
        for j in range(len(lmbd)):
            neur = NNdata[i][j]
            yPredTrain = neur.predict(XTrain)
            yProbTrain = neur.predict_proba(XTrain)

            yPredTest = neur.predict(XTest)
            yProbTest = neur.predict_proba(XTest)

            scoresArea[i][j] = neur.get_Area_ratio(np.ravel(yTest), yProbTest)
            scoresF1[i][j] = f1_score(yTest, np.round(yPredTest))
            scoresAUC[i][j] = roc_auc_score(yTest, yPredTest)

    mx, my = np.unravel_index(np.argmax(scoresArea + scoresF1 + scoresAUC), scoresArea.shape)

    sns.set()
    fig, ax = plt.subplots(figsize = (7, 7))
    sns.heatmap(scoresF1, annot=True, fmt=".3f",ax=ax, cmap="viridis", xticklabels=lmbd, yticklabels=eta, cbar=False)

    ax.set_xlabel("$\lambda$", fontsize = 12)
    ax.set_ylabel("$\eta$", fontsize=12)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    #plt.savefig(figpath + "NN_credit" + act + ".pdf")
    #plt.show()

    file = open(respath + act + "_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % scoresAUC[mx][my])
    file.write("Sklearn F1 = %1.4f\n" % scoresF1[mx][my])
    file.write("Test area ratio = %1.4f\n" % scoresArea[mx][my])
    file.close()




if comp_sklearn:
    test, train = get_credit_data()
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    n_hid_neur = 50; n_epochs=100; eta=0.1; lmbd=0.1

    dnn = MLPClassifier(hidden_layer_sizes=n_hid_neur, activation='logistic',
    alpha=lmbd, learning_rate_init=eta, max_iter=n_epochs)
    dnn.fit(XTrain, yTrain_onehot)

    z = np.argmax(dnn.predict_proba(XTrain), axis=1)

    plt.hist(z)
    plt.show()
    print(dnn.score(XTest, yTest_onehot))
    print(dnn.score(XTrain, yTrain_onehot))

if comp_franke:
    #np.random.seed(42)
    def R2(z, z_tilde):
        RR_res = np.sum((z - z_tilde)**2)
        RR_tot = np.sum((z - np.mean(z))**2)
        return 1 - RR_res/RR_tot

    def MSE(z, z_tilde):
        return np.mean((z - z_tilde)**2)


    n = 100; p = 5
    #x, y, z = get_train_data(n, noise=True)
    #X = CreateDesignMatrix(x, y, p)
    np.random.seed(42)
    test, train = get_test_train_data(n, 0.5, False)
    xtrain, ytrain, zTrain = train
    xtest, ytest, zTest = test

    XTrain = CreateDesignMatrix(xtrain, ytrain, p)
    XTest = CreateDesignMatrix(xtest, ytest, p)

    zTrain = zTrain.reshape(-1,1)

    eta = np.logspace(-4, 0, 5)
    lmbd = np.logspace(-8, -4, 5)
    scoresTrain = np.zeros((2, len(eta), len(lmbd)))
    scoresTest = scoresTrain.copy()

    NNdata = np.zeros((len(eta), len(lmbd)), dtype=object)
    samp = ""

    run = True
    if run:
        for i, e in enumerate(tqdm(eta)):
            for j, l in enumerate(lmbd):
                neurLin = NeuralLinReg(XTrain, zTrain,
                eta=e,
                lmbd=l,
                hid_layers=1,
                n_epochs=100,
                b_size=50)
                neurLin.train()
                NNdata[i][j] = neurLin

        np.save(datapath + "NN_data_franke" + samp, NNdata)

    else:
        NNdata = np.load(datapath + "NN_data_franke" + samp + ".npy")

    for i in range(len(eta)):
        for j in range(len(lmbd)):
            neurLin = NNdata[i][j]
            ztildeTrain = neurLin.feed_forward_out(XTrain)
            scoresTrain[0][i][j] = R2(np.ravel(zTrain), np.ravel(ztildeTrain))
            scoresTrain[1][i][j] = MSE(np.ravel(zTrain), np.ravel(ztildeTrain))

            ztildeTest = neurLin.feed_forward_out(XTest)
            scoresTest[0][i][j] = R2(np.ravel(zTest), np.ravel(ztildeTest))
            scoresTest[1][i][j] = MSE(np.ravel(zTest), np.ravel(ztildeTest))


    sns.set()

    fig, ax = plt.subplots(1, 2, figsize = (15, 10))
    sns.heatmap(scoresTest[0], annot=True, ax=ax[0], cmap="viridis", xticklabels=lmbd, yticklabels=eta)
    sns.heatmap(scoresTest[1], annot=True, ax=ax[1], cmap="viridis", xticklabels=lmbd, yticklabels=eta)
    ax[0].set_title("Test Accuracy R2")
    ax[1].set_title("Test Accuracy MSE")
    for i in range(2):
        ax[i].set_ylabel("$\eta$")
        ax[i].set_xlabel("$\lambda$")
        #ax[i].set_xticklabels(lmbd)
        #ax[i].set_yticklabels(eta)
        #ax[i].ticklabel_format(axis='both',style='sci')



    plt.show()

    fig, ax = plt.subplots(1, 2, figsize = (15, 10))
    sns.heatmap(scoresTrain[0], annot=True, ax=ax[0], cmap="viridis", xticklabels=lmbd, yticklabels=eta)
    sns.heatmap(scoresTrain[1], annot=True, ax=ax[1], cmap="viridis", xticklabels=lmbd, yticklabels=eta)
    ax[0].set_title("Train Accuracy R2")
    ax[1].set_title("Train Accuracy MSE")
    ax[0].set_ylabel("$\eta$")
    ax[0].set_xlabel("$\lambda$")
    ax[1].set_ylabel("$\eta$")
    ax[1].set_xlabel("$\lambda$")
    plt.show()
