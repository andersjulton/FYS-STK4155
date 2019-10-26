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
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings('ignore')

figpath = "figures/NN/"
datapath = "datafiles/"
respath = "results/NN/"


comp_credit = False
comp_sklearn = False
comp_franke = False

def get_credit_data(change_values=True, remove_values=False, up_sample=False, down_sample=False):

    X, y = readfile(change_values=change_values, remove_values=remove_values)

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

if comp_credit:
    test, train = get_credit_data(change_values=True, remove_values=False, down_sample=True)
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    eta = np.logspace(-6, -1, 6)
    lmbd = np.logspace(-6, -1, 6)

    scores = np.zeros((2, len(eta), len(lmbd)))
    scoresArea = np.zeros((2, len(eta), len(lmbd)))
    NNdata = np.zeros((len(eta), len(lmbd)), dtype=object)

    hid_AF = ActivationFunctions().PReLU
    out_AF = ActivationFunctions().sigmoid

    run = True
    act = "_PReLU"

    if run:
        for i, e in enumerate(tqdm(eta)):
            for j, l in enumerate(lmbd):
                neur = NeuralLogReg(XTrain, yTrain_onehot,
                    hid_method=hid_AF,
                    out_method=out_AF,
                    n_hid_neur=50,
                    n_cat=2,
                    n_epochs=40,
                    b_size=500,
                    eta=e,
                    lmbd=l,
                    hid_layers=2)
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

            scores[0][i][j] = (yPredTest == yTest).mean()
            scoresArea[0][i][j] = neur.get_Area_ratio(np.ravel(yTest), yProbTest)
            scores[1][i][j] = (yPredTrain == yTrain).mean()
            scoresArea[1][i][j] = neur.get_Area_ratio(np.ravel(yTrain), yProbTrain)

    mxr, myr = np.unravel_index(np.argmax(scoresArea[0]), scoresArea[0].shape)
    mx, my = np.unravel_index(np.argmax(scores[0]), scores[0].shape)

    file = open(respath + "NN_results" + act +".txt", "w+")

    file.write("Test score = %1.3f, Lambda = %e, Eta = %e\n" % (np.max(scores[0]), lmbd[my], eta[mx]))
    file.write("Train score = %1.3f\n" % np.max(scores[1]))
    file.write("Test area ratio = %1.3f, Lambda = %e, Eta = %e\n" % (np.max(scoresArea[0]), lmbd[myr], eta[mxr]))
    file.write("Train area ratio = %1.3f\n" % np.max(scoresArea[1]))
    file.close()



    sns.set()
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    sns.heatmap(scoresArea[0], annot=True, ax=ax[0,0], cmap="viridis", xticklabels=lmbd, yticklabels=eta, cbar=False)
    sns.heatmap(scoresArea[1], annot=True, ax=ax[0,1], cmap="viridis", xticklabels=lmbd, yticklabels=eta, cbar=False)
    sns.heatmap(scores[0], annot=True, ax=ax[1,0], cmap="viridis", xticklabels=lmbd, yticklabels=eta, cbar=False)
    sns.heatmap(scores[1], annot=True, ax=ax[1,1], cmap="viridis", xticklabels=lmbd, yticklabels=eta, cbar=False)
    ax[0,0].set_title("Test Area ratio")
    ax[0,1].set_title("Training Area ratio")
    ax[1,0].set_title("Test Accuracy")
    ax[1,1].set_title("Training accuray")

    for i in range(2):
        ax[1][i].set_xlabel("$\lambda$")
        ax[i][0].set_ylabel("$\eta$")
        for j in range(2):
            bottom, top = ax[i][j].get_ylim()
            ax[i][j].set_ylim(bottom + 0.5, top - 0.5)


    plt.savefig(figpath + "NN_credit" + act + ".pdf")
    plt.show()




if comp_sklearn:
    test, train = get_credit_data(True, False)
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
                neurLin = NeuralLinReg(XTrain, zTrain, n_cat=1, eta=e, lmbd=l, hid_layers=1, n_epochs=100, b_size=50)
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
