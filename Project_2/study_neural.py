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


grid_search_credit = False
comp_sklearn = False
grid_search_franke = False
comp_franke = True

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

    hid_AF = ActivationFunctions().tanh
    out_AF = ActivationFunctions().sigmoid

    run = True
    act = "_tanh"

    if run:
        for i, e in enumerate(tqdm(eta)):
            for j, l in enumerate(lmbd):
                neur = NeuralLogReg(XTrain, yTrain_onehot,
                    hid_actFunc=hid_AF,
                    out_actFunc=out_AF,
                    n_epochs=50,
                    b_size=500,
                    eta=e,
                    lmbd=l,
                    n_hid_neur = [5, 10],
                    hid_layers=2)
                neur.train()
                NNdata[i][j] = neur

        #np.save(datapath + "NN_data_credit" + act, NNdata)

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
    sns.heatmap(scoresArea, annot=True, fmt=".3f",ax=ax, cmap="viridis", xticklabels=lmbd, yticklabels=eta, cbar=False)

    ax.set_xlabel("$\lambda$", fontsize = 12)
    ax.set_ylabel("$\eta$", fontsize=12)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(figpath + "NN_credit" + act + ".pdf")
    plt.show()

    file = open(respath + act + "_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % scoresAUC[mx][my])
    file.write("Sklearn F1 = %1.4f\n" % scoresF1[mx][my])
    file.write("Test area ratio = %1.4f\n" % scoresArea[mx][my])
    file.write("Lambda = %e, eta = %e" %(lmbd[mu], eta[mx]))
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

if grid_search_franke:
    np.random.seed(42)




    n = 100; p = 5
    #x, y, z = get_train_data(n, noise=True)
    #X = CreateDesignMatrix(x, y, p)
    #np.random.seed(42)
    xtrain, ytrain, zTrain = get_train_data(n, True)
    xtest, ytest, zTest = get_train_data(n, False)

    XTrain = CreateDesignMatrix(xtrain, ytrain, p)
    XTest = CreateDesignMatrix(xtest, ytest, p)

    zTrain = zTrain.reshape(-1,1)

    eta = np.logspace(-3, -3, 1)
    lmbd = np.logspace(-3, -3, 1)
    hid_AF = ActivationFunctions().PReLU
    out_AF = ActivationFunctions().PReLU

    scoresTrain = np.zeros((2, len(eta), len(lmbd)))
    scoresTest = scoresTrain.copy()
    zdata = np.zeros((len(eta), len(lmbd), len(zTest)))

    NNdata = np.zeros((len(eta), len(lmbd)), dtype=object)
    act = ""

    w_sigma = [np.sqrt(2/(XTrain.shape[1] + 100)), np.sqrt(2/(200 + 100)), np.sqrt(2/201)]

    run = True
    if run:
        for i, e in enumerate(tqdm(eta)):
            for j, l in enumerate(lmbd):
                neurLin = NeuralLinReg(XTrain, zTrain,
                eta=e,
                lmbd=0,
                out_actFunc = out_AF,
                hid_actFunc = hid_AF,
                hid_layers= 2,
                n_hid_neur= [100, 200],
                n_epochs=100,
                b_size=100,
                init_bias = 0,
                w_sigma = w_sigma)
                neurLin.train()
                NNdata[i][j] = neurLin

        #np.save(datapath + "NN_data_franke" + act, NNdata)

    else:
        NNdata = np.load(datapath + "NN_data_franke" + act + ".npy")

    for i in range(len(eta)):
        for j in range(len(lmbd)):
            neurLin = NNdata[i][j]
            ztildeTrain = neurLin.feed_forward_out(XTrain)
            scoresTrain[0][i][j] = neurLin.R2(np.ravel(zTrain), np.ravel(ztildeTrain))
            scoresTrain[1][i][j] = neurLin.MSE(np.ravel(zTrain), np.ravel(ztildeTrain))

            ztildeTest = neurLin.feed_forward_out(XTest)
            scoresTest[0][i][j] = neurLin.R2(np.ravel(zTest), np.ravel(ztildeTest))
            scoresTest[1][i][j] = neurLin.MSE(np.ravel(zTest), np.ravel(ztildeTest))
            zdata[i][j] = np.ravel(ztildeTest)

    mx, my = np.unravel_index(np.nanargmax(scoresTest[0]), scoresTest[0].shape)


    x = xtest.reshape(n, n)
    y = ytest.reshape(n, n)
    z = zdata[mx][my].reshape(n, n)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap="coolwarm", linewidth=0, antialiased=False, alpha=0.5)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 60)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_zlabel(r"$f(x, y)$", fontsize=13)
    plt.tight_layout()
	#plt.savefig("figures/" + filename + "franke.pdf")
    plt.show()


    sns.set()

    fig, ax = plt.subplots(1, 2, figsize = (7, 7))
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

    fig, ax = plt.subplots(1, 2, figsize = (7, 7))
    sns.heatmap(scoresTrain[0], annot=True, ax=ax[0], cmap="viridis", xticklabels=lmbd, yticklabels=eta)
    sns.heatmap(scoresTrain[1], annot=True, ax=ax[1], cmap="viridis", xticklabels=lmbd, yticklabels=eta)
    ax[0].set_title("Train Accuracy R2")
    ax[1].set_title("Train Accuracy MSE")
    ax[0].set_ylabel("$\eta$")
    ax[0].set_xlabel("$\lambda$")
    ax[1].set_ylabel("$\eta$")
    ax[1].set_xlabel("$\lambda$")
    plt.show()

if comp_franke:
    np.random.seed(42)
    n = 100; p = 5

    xtrain, ytrain, zTrain = get_train_data(n, True)
    xtest, ytest, zTest = get_train_data(n, False)

    XTrain = CreateDesignMatrix(xtrain, ytrain, p)
    XTest = CreateDesignMatrix(xtest, ytest, p)

    zTrain = zTrain.reshape(-1,1)

    hid_AF = ActivationFunctions(a=0.1).ReLU
    out_AF = ActivationFunctions(a=0.1).ReLU

    act = "_ReLU"

    hneur = [100, 100, 100, 100]

    w_sigma = [np.sqrt(2/(XTrain.shape[1] + hneur[0])),
        np.sqrt(2/(hneur[1] + hneur[0])),
        np.sqrt(2/(hneur[2] + hneur[1])),
        np.sqrt(2/(hneur[2] + hneur[3])),
        np.sqrt(2/hneur[1] + 1)]

    neurLin = NeuralLinReg(XTrain, zTrain,
        eta = 0.00001,
        lmbd = 0,
        out_actFunc = out_AF,
        hid_actFunc = hid_AF,
        hid_layers = 4,
        n_hid_neur = hneur,
        n_epochs = 100,
        b_size = 500,
        init_bias = 0.01,
        w_sigma = w_sigma)
    neurLin.train()

    ztildeTest = neurLin.feed_forward_out(XTest)

    print("R2: ", neurLin.R2(np.ravel(zTest), np.ravel(ztildeTest)))
    print("MSE: ", neurLin.MSE(np.ravel(zTest), np.ravel(ztildeTest)))

    x = xtest.reshape(n, n)
    y = ytest.reshape(n, n)
    z = ztildeTest.reshape(n, n)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 60)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_zlabel(r"$f(x, y)$", fontsize=13)
    plt.tight_layout()
    plt.savefig(figpath + act + "_3D_franke.pdf")
    plt.show()

    file = open(respath + "NN_franke" + act + "_results.txt", "w+")

    file.write("R2= %1.4f\n" % neurLin.R2(np.ravel(zTest), np.ravel(ztildeTest)))
    file.write("MSE = %1.4f\n" % neurLin.MSE(np.ravel(zTest), np.ravel(ztildeTest)))

    file.close()
