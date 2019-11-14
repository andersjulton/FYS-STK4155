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
neuron_search_credit = False
epoch_search_credit = False
random_search_credit = False
comp_sklearn = False
grid_search_franke = False
comp_franke = False
comp_bias_var_franke = True

def get_credit_data(up_sample=False, down_sample=False):

    X, y = readfile()

    trainingShare = 0.5
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

if random_search_credit:
    test, train = get_credit_data(down_sample=True)
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    m = 100
    NNdata = np.zeros(m, dtype=object)
    scoresArea = np.zeros(m)

    hid_AF = ActivationFunctions().sigmoid
    out_AF = ActivationFunctions().sigmoid

    run = False
    if run:
        for i in tqdm(range(m)):
            NNLogReg = NeuralLogReg(XTrain, yTrain_onehot,
                hid_actFunc=hid_AF,
                out_actFunc=out_AF,
                n_epochs=50,
                b_size=200,
                eta=0.01,
                lmbd=0.1,
                n_hid_neur = 20,
                hid_layers=1)
            NNLogReg.train()
            NNdata[i] = NNLogReg

        np.save(datapath + "random_search_credit", NNdata)

    else:
        NNdata = np.load(datapath + "random_search_credit.npy", allow_pickle=True)

    for i in range(m):
        NNLogReg = NNdata[i]

        yProbTest = NNLogReg.predict_proba(XTest)
        scoresArea[i] = NNLogReg.get_Area_ratio(np.ravel(yTest), yProbTest)

    print(np.std(scoresArea))
    plt.hist(scoresArea, bins=20, edgecolor='black', facecolor='lightblue')
    plt.axvline(np.mean(scoresArea), color='r', linestyle='--', label="Mean = {:.3}".format(np.mean(scoresArea)))
    plt.xlabel("Area ratio score", fontsize=14)
    plt.ylabel("Number of runs in bin", fontsize=14)
    plt.tight_layout()
    plt.tick_params('both', labelsize=12)
    plt.legend(fontsize=12)
    plt.savefig(figpath + "random_search.pdf")
    plt.show()


if epoch_search_credit:
    test, train = get_credit_data(down_sample=True)
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    epochs = np.arange(5, 105, 5)
    NNdata = np.zeros((len(epochs), 5), dtype=object)
    scoresArea = np.zeros(len(epochs))
    scoresF1 = scoresArea.copy()

    hid_AF = ActivationFunctions().sigmoid
    out_AF = ActivationFunctions().sigmoid

    run = True
    if run:
        for i, e in enumerate(tqdm(epochs)):
            for j in range(5):
                NNLogReg = NeuralLogReg(XTrain, yTrain_onehot,
                    hid_actFunc=hid_AF,
                    out_actFunc=out_AF,
                    n_epochs=e,
                    b_size=200,
                    eta=0.01,
                    lmbd=0.1,
                    n_hid_neur = 20,
                    hid_layers=1)
                NNLogReg.train()
                NNdata[i][j] = NNLogReg

        #np.save(datapath + "epoch_search_credit", NNdata)

    else:
        NNdata = np.load(datapath + "epoch_search_credit.npy", allow_pickle=True)

    for i in range(len(epochs)):
        areaScore = 0
        F1score = 0
        for j in range(5):
            NNLogReg = NNdata[i][j]

            yPredTest = NNLogReg.predict(XTest)
            yProbTest = NNLogReg.predict_proba(XTest)

            areaScore += NNLogReg.get_Area_ratio(np.ravel(yTest), yProbTest)
            F1score += f1_score(yTest, np.round(yPredTest))

        scoresArea[i] = areaScore/5
        scoresF1[i] = F1score/5

    plt.plot(epochs, scoresArea, label="Area ratio")
    plt.plot(epochs, scoresF1, label="F1")
    plt.xlabel("Number of epochs", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.tick_params('both', labelsize=12)
    #plt.savefig(figpath + "epoch_search.pdf")
    plt.show()

if neuron_search_credit:
    test, train = get_credit_data(down_sample=True)
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    n_neur = np.arange(10, 210, 10)
    NNdata = np.zeros((len(n_neur), 5), dtype=object)
    scoresArea = np.zeros(len(n_neur))
    scoresF1 = scoresArea.copy()

    hid_AF = ActivationFunctions().sigmoid
    out_AF = ActivationFunctions().sigmoid

    run = False
    if run:
        for i, n in enumerate(tqdm(n_neur)):
            for j in range(5):
                NNLogReg = NeuralLogReg(XTrain, yTrain_onehot,
                    hid_actFunc=hid_AF,
                    out_actFunc=out_AF,
                    n_epochs=100,
                    b_size=200,
                    eta=0.01,
                    lmbd=0.1,
                    n_hid_neur = n,
                    hid_layers=1)
                NNLogReg.train()
                NNdata[i][j] = NNLogReg

        np.save(datapath + "neur_search_credit", NNdata)

    else:
        NNdata = np.load(datapath + "neur_search_credit.npy", allow_pickle=True)

    for i in range(len(n_neur)):
        areaScore = 0
        F1score = 0
        for j in range(5):
            NNLogReg = NNdata[i][j]

            yPredTest = NNLogReg.predict(XTest)
            yProbTest = NNLogReg.predict_proba(XTest)

            areaScore += NNLogReg.get_Area_ratio(np.ravel(yTest), yProbTest)
            F1score += f1_score(yTest, np.round(yPredTest))

        scoresArea[i] = areaScore/5
        scoresF1[i] = F1score/5

    plt.plot(n_neur, scoresArea, label="Area ratio")
    #plt.plot(n_neur, scoresF1, label="F1")
    plt.xlabel("Number of hidden neurons", fontsize=14)
    plt.ylabel("Area ratio", fontsize=14)
    #plt.legend(fontsize=14)
    plt.tight_layout()
    plt.tick_params('both', labelsize=12)
    plt.savefig(figpath + "hidden_neuron_search.pdf")
    plt.show()

if grid_search_credit:
    test, train = get_credit_data(down_sample=True)
    XTest, yTest, yTest_onehot = test
    XTrain, yTrain, yTrain_onehot = train

    eta = np.logspace(-6, -1, 6)
    lmbd = np.logspace(-1, -1, 1)

    scoresArea = np.zeros((len(eta), len(lmbd)))
    scoresF1 = scoresArea.copy()
    scoresAUC = scoresArea.copy()

    NNdata = np.zeros((len(eta), len(lmbd)), dtype=object)

    hid_AF = ActivationFunctions().sigmoid
    out_AF = ActivationFunctions().sigmoid

    run = True
    act = "_sigmoid"

    if run:
        for i, e in enumerate(tqdm(eta)):
            for j, l in enumerate(lmbd):
                neur = NeuralLogReg(XTrain, yTrain_onehot,
                    hid_actFunc=hid_AF,
                    out_actFunc=out_AF,
                    n_epochs=100,
                    b_size=200,
                    eta=e,
                    lmbd=l,
                    n_hid_neur = [5, 10],
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

            scoresArea[i][j] = neur.get_Area_ratio(np.ravel(yTest), yProbTest)
            scoresF1[i][j] = f1_score(yTest, np.round(yPredTest))
            scoresAUC[i][j] = roc_auc_score(yTest, yPredTest)

    mx, my = np.unravel_index(np.argmax(scoresArea + scoresF1 + scoresAUC), scoresArea.shape)

    sns.set()
    fig, ax = plt.subplots(figsize = (7, 7))
    sns.heatmap(scoresArea, annot=True, fmt=".3f",ax=ax, cmap="viridis", xticklabels=lmbd, yticklabels=eta, cbar=False)

    ax.set_xlabel("$\lambda$", fontsize = 14)
    ax.set_ylabel("$\eta$", fontsize=14)
    ax.tick_params(labelsize=14)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.tight_layout()
    plt.savefig(figpath + "NN_credit" + act + ".pdf")
    plt.show()

    file = open(respath + act + "_results.txt", "w+")

    file.write("Sklearn AUC Score = %1.4f\n" % scoresAUC[mx][my])
    file.write("Sklearn F1 = %1.4f\n" % scoresF1[mx][my])
    file.write("Test area ratio = %1.4f\n" % scoresArea[mx][my])
    file.write("Lambda = %e, eta = %e" %(lmbd[my], eta[mx]))
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

    xtrain, ytrain, zTrain = get_train_data(n, True)
    xtest, ytest, zTest = get_train_data(n, False)

    XTrain = CreateDesignMatrix(xtrain, ytrain, p)
    XTest = CreateDesignMatrix(xtest, ytest, p)

    zTrain = zTrain.reshape(-1, 1)

    eta = np.logspace(-6, -1, 6)
    lmbd = np.logspace(-1, -1, 1)
    hid_AF = ActivationFunctions(a = 0.1).ELU
    out_AF = ActivationFunctions().ReLU

    scoresTrain = np.zeros((2, len(eta), len(lmbd)))
    scoresTest = scoresTrain.copy()
    zdata = np.zeros((len(eta), len(lmbd), len(zTest)))

    NNdata = np.zeros((len(eta), len(lmbd)), dtype=object)
    act = "ELU_1_layer"

    hneur = 50#[100, 100, 100, 100]
    w_sigma = np.sqrt(2/(XTrain.shape[1] + hneur))
    """
    w_sigma = [np.sqrt(2/(XTrain.shape[1] + hneur[0])),
    np.sqrt(2/(hneur[1] + hneur[0])),
    np.sqrt(2/(hneur[2] + hneur[1])),
    np.sqrt(2/(hneur[2] + hneur[3])),
    np.sqrt(2/hneur[1] + 1)]"""

    run = True
    if run:
        for i, e in enumerate(tqdm(eta)):
            for j, l in enumerate(lmbd):
                neurLin = NeuralLinReg(XTrain, zTrain,
                eta=e,
                lmbd=l,
                out_actFunc = out_AF,
                hid_actFunc = hid_AF,
                hid_layers= 1,
                n_hid_neur= hneur,
                n_epochs=500,
                b_size=500,
                init_bias = 0.01,
                w_sigma = w_sigma)
                neurLin.train()
                NNdata[i][j] = neurLin

        np.save(datapath + "NN_data_franke" + act, NNdata)

    else:
        NNdata = np.load(datapath + "NN_data_franke" + act + ".npy")

    for i in range(len(eta)):
        for j in range(len(lmbd)):
            neurLin = NNdata[i][j]

            ztildeTest = neurLin.feed_forward_out(XTest)
            scoresTest[0][i][j] = neurLin.R2(np.ravel(zTest), np.ravel(ztildeTest))
            scoresTest[1][i][j] = neurLin.MSE(np.ravel(zTest), np.ravel(ztildeTest))
            zdata[i][j] = np.ravel(ztildeTest)

    mx, my = np.unravel_index(np.nanargmax(scoresTest[0]), scoresTest[0].shape)

    sns.set()

    fig, ax = plt.subplots(figsize = (7, 7))
    sns.heatmap(scoresTest[0], annot=True, ax=ax, cmap="viridis", xticklabels=lmbd, yticklabels=eta)

    ax.set_xlabel("$\lambda$", fontsize = 14)
    ax.set_ylabel("$\eta$", fontsize=14)
    ax.tick_params(labelsize=14)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.tight_layout()
    plt.savefig(figpath + "NN_franke" + act + ".pdf")
    plt.show()



if comp_franke:
    np.random.seed(42)
    n = 100; p = 5

    xtrain, ytrain, zTrain = get_train_data(n, True)
    xtest, ytest, zTest = get_train_data(n, False)

    XTrain = CreateDesignMatrix(xtrain, ytrain, p)
    XTest = CreateDesignMatrix(xtest, ytest, p)


    zTrain = zTrain.reshape(-1,1)

    hid_AF = ActivationFunctions(a=0.05).ReLU
    out_AF = ActivationFunctions(a=0.05).ReLU

    act = "_ReLU"

    hneur = 100#[100, 100, 100, 100]
    w_sigma = [np.sqrt(2/(XTrain.shape[1] + hneur[0]))]
    """w_sigma = [np.sqrt(2/(XTrain.shape[1] + hneur[0])),
        np.sqrt(2/(hneur[1] + hneur[0])),
        np.sqrt(2/(hneur[2] + hneur[1])),
        np.sqrt(2/(hneur[2] + hneur[3])),
        np.sqrt(2/hneur[1] + 1)]"""

    neurLin = NeuralLinReg(XTrain, zTrain,
        eta = 0.00001,
        lmbd = 0,
        out_actFunc = out_AF,
        hid_actFunc = hid_AF,
        hid_layers = 1,
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
    surf = ax.plot_surface(x, y, zTest.reshape(n,n), cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 60)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_zlabel(r"$f(x, y)$", fontsize=13)
    plt.tight_layout()
    plt.savefig(figpath + act + "_3D_franke_test.pdf")
    plt.show()

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
    plt.savefig(figpath + act + "_3D_franke_fit.pdf")
    plt.show()

    zTrain = np.ravel(zTrain)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xtrain.reshape(n,n), y.reshape(n,n), zTrain.reshape(n,n), cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(30, 60)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_zlabel(r"$f(x, y)$", fontsize=13)
    plt.tight_layout()
    plt.savefig(figpath + act + "_3D_franke_train.pdf")
    plt.show()

    file = open(respath + "NN_franke" + act + "_results.txt", "w+")

    file.write("R2= %1.4f\n" % neurLin.R2(np.ravel(zTest), np.ravel(ztildeTest)))
    file.write("MSE = %1.4f\n" % neurLin.MSE(np.ravel(zTest), np.ravel(ztildeTest)))

    file.close()

if comp_bias_var_franke:
    #np.random.seed(42)

    hid_AF = ActivationFunctions(a=0.05).ReLU
    out_AF = ActivationFunctions(a=0.05).ReLU
    hneur = [100]
    n = 100; p = 5
    epochs = np.arange(10, 220, 20)

    MSEscores = np.zeros((2, len(epochs)))

    run = False
    if run:
        for i, e in enumerate(tqdm(epochs)):
            trainscore = 0
            testscore = 0
            for j in range(5):

                xtrain, ytrain, zTrain = get_train_data(n, True)
                xtest, ytest, zTest = get_train_data(n, True)

                XTrain = CreateDesignMatrix(xtrain, ytrain, p)
                XTest = CreateDesignMatrix(xtest, ytest, p)

                zTrain = zTrain.reshape(-1,1)

                w_sigma = [np.sqrt(2/(XTrain.shape[1] + hneur[0]))]


                neurLin = NeuralLinReg(XTrain, zTrain,
                    eta = 0.00001,
                    lmbd = 0,
                    out_actFunc = out_AF,
                    hid_actFunc = hid_AF,
                    hid_layers = 1,
                    n_hid_neur = hneur,
                    n_epochs = e,
                    b_size = 500,
                    init_bias = 0.01,
                    w_sigma = w_sigma)
                neurLin.train()

                ztildeTest = neurLin.feed_forward_out(XTest)
                ztildeTrain = neurLin.feed_forward_out(XTrain)

                testscore += neurLin.MSE(np.ravel(zTest), np.ravel(ztildeTest))
                trainscore +=  neurLin.MSE(np.ravel(zTrain), np.ravel(ztildeTrain))


            MSEscores[0][i] = testscore/5
            MSEscores[1][i] = trainscore/5

        np.save(datapath + "bias_var_franke_epochs", MSEscores)

    else:
        MSEscores = np.load(datapath + "bias_var_franke_epochs.npy")


    plt.plot(epochs, MSEscores[0], label='Test sample', color = "mediumblue")
    plt.plot(epochs, MSEscores[1], label='Training sample', color = "crimson")
    plt.xlabel("Number of epochs", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(figpath + "bias_var_franke_epochs.pdf")
    plt.show()
