from franke import *
from regClass import OLS, LASSO, RIDGE
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import tqdm
from sklearn.preprocessing import normalize

"""
Good seeds for testing:
Ridge: Seed = 42, n = 81, M(lambdas) = 100

"""

#np.random.seed(42)

# I AM SO SORRY
import warnings
warnings.filterwarnings('ignore')

fsize = 10				# universal fontsize for plots
path = "figures/"

def plot_OLS():
	"""
	OLS as function of n, mean over 10 runs. Tested against Franke's without noise.
	"""

	ols = OLS(5)
	N = np.linspace(50, 150, 101, dtype="int")
	m = np.linspace(0, 9, 10, dtype="int")
	MSEtrain, R2train = np.zeros((10, 101)), np.zeros((10, 101))
	MSEtest, R2test = np.zeros((10, 101)), np.zeros((10, 101))

	for i in tqdm.tqdm(m):
		for j, n in enumerate(N):
			test, train = get_test_train_data(n, 0.2, False)
			xtest, ytest, ztest = test
			xtrain, ytrain, ztrain = train

			Xtrain = ols.CreateDesignMatrix(xtrain, ytrain)
			Xtest = ols.CreateDesignMatrix(xtest, ytest)

			ols.fit(Xtrain, ztrain)

			MSEtest[i][j] = ols.MSE(ztest, ols(Xtest))
			R2test[i][j] = ols.R2(ztest, ols(Xtest))

			MSEtrain[i][j] = ols.MSE(ztrain, ols(Xtrain))
			R2train[i][j] = ols.R2(ztrain, ols(Xtrain))

	plt.plot(N, np.mean(MSEtest, axis=0), label="Test data")
	plt.plot(N, np.mean(MSEtrain, axis=0), label="Training data", color='red')
	plt.ylabel("MSE", fontsize=fsize)
	plt.xlabel("n", fontsize=fsize)
	plt.legend()
	#plt.savefig(path + "OLS(n)_MSE.pdf")
	plt.show()

	plt.plot(N, np.mean(R2test, axis=0), label="Test data")
	#plt.plot(N, np.mean(R2train, axis=0), label="Training data")
	plt.ylabel("R2", fontsize=fsize)
	plt.xlabel("n", fontsize=fsize)
	plt.legend()

	#plt.savefig(path + "OLS(n)_R2.pdf")
	plt.show()

if True:
	plot_OLS()

def plot_MSE_R2(n, noise):
	"""
	Ridge and LASSO as function of lambda.
	"""
	p = 5
	ridge = RIDGE(p, 0)
	lasso = LASSO(p, 0)

	M = 100
	lambdasR = np.logspace(-6, -2, M)
	lambdasL = np.logspace(-6, -2, M)
	lambdas = [lambdasR, lambdasL]
	MSE = np.zeros((2, M))
	R2 = np.zeros((2, M))

	test, train = get_test_train_data(n, test_size=0.2, noise=noise)
	xtrain, ytrain, ztrain = train
	xtest, ytest, ztest = test
	Xtrain = ridge.CreateDesignMatrix(xtrain, ytrain)
	Xtest = ridge.CreateDesignMatrix(xtest, ytest)

	for i in range(M):
		ridge.l = lambdas[0][i]
		lasso.l = lambdas[1][i]

		ridge.fit(Xtrain, ztrain)
		MSE[0][i] = ridge.MSE(ztest, ridge(Xtest))
		R2[0][i] = ridge.R2(ztest, ridge(Xtest))

		lasso.fit(Xtrain, ztrain)
		MSE[1][i] = lasso.MSE(ztest, lasso(Xtest))
		R2[1][i] = lasso.R2(ztest, lasso(Xtest))

	labels = ["RIDGE", "LASSO"]

	for i in range(2):
		title = " for n = %d " %n
		if noise:
			end = str(n) + "_noise.pdf"
			title += "with noise"
		else:
			end = str(n) + "_.pdf"
			title += "without noise"
		minMSE = np.argmin(MSE[i])
		maxR2 = np.argmax(R2[i])

		MSEind = np.linspace(minMSE - 30, minMSE + 30, 61, dtype="int16")
		R2ind = np.linspace(maxR2 - 30, maxR2 + 30, 61, dtype="int16")
		plt.semilogx(lambdas[i], MSE[i], label=labels[i])
		plt.scatter(lambdas[i][minMSE], MSE[i][minMSE], color='r', label=r"$\lambda = %1.5f$" % lambdas[i][minMSE])
		plt.title("MSE" + title)
		plt.legend(fontsize=fsize)
		plt.xlabel(r"$log_{10}(\lambda)$", fontsize=fsize)
		plt.ylabel("MSE", fontsize=fsize)
		plt.tight_layout()
		#plt.savefig(path + "MSE_methods_" + labels[i] + "_" + end)
		plt.show()

		plt.semilogx(lambdas[i], R2[i], label=labels[i])
		plt.scatter(lambdas[i][maxR2], R2[i][maxR2], color='r', label=r"$\lambda = %1.5f$" % lambdas[i][maxR2])
		plt.title(r"$R^2$" + title)
		plt.legend(fontsize=fsize)
		plt.xlabel(r"$log_{10}(\lambda)$", fontsize=fsize)
		plt.ylabel("$R^2$", fontsize=fsize)
		plt.tight_layout()
		#plt.savefig(path + "R2_methods_" + labels[i] + "_" + end)
		plt.show()

if False:
	n = 50
	plot_MSE_R2(n, True)


def plot_lamb_poly(n, noise):
	M = 100
	lambdas = np.logspace(-7, -2, M)

	polys = np.arange(4, 16)
	MSER = np.zeros((M, len(polys)))
	R2R = MSER.copy()
	MSEL = MSER.copy()
	R2L = MSER.copy()
	test, train = get_test_train_data(n, test_size=0.20, noise=noise)
	xtrain, ytrain, ztrain = train
	xtest, ytest, ztest = test

	for i, p in enumerate(tqdm.tqdm(polys)):
		ridge = RIDGE(p, 0)
		lasso = LASSO(p, 0)

		Xtrain = ridge.CreateDesignMatrix(xtrain, ytrain)
		Xtest = ridge.CreateDesignMatrix(xtest, ytest)

		for j in range(M):
			ridge.l = lambdas[j]
			lasso.l = lambdas[j]

			ridge.fit(Xtrain, ztrain)
			MSER[j][i] = ridge.MSE(ztest, ridge(Xtest))
			R2R[j][i] = ridge.R2(ztest, ridge(Xtest))

			lasso.fit(Xtrain, ztrain)
			MSEL[j][i] = lasso.MSE(ztest, lasso(Xtest))
			R2L[j][i] = lasso.R2(ztest, lasso(Xtest))

	pointsR = np.argmin(MSER, axis=0)
	plt.scatter(polys, lambdas[pointsR])
	plt.ylim([0, np.max(lambdas[pointsR]) + 0.1*np.max(lambdas[pointsR])])
	plt.title("Best lambda(p) for Ridge")
	plt.xlabel("Polynomial degree")
	plt.ylabel(r"$\lambda$")
	plt.savefig(path + "best_lambda_RIDGE.pdf")
	plt.show()

	pointsL = np.argmin(MSEL, axis=0)
	plt.scatter(polys, lambdas[pointsL])
	plt.ylim([0, np.max(lambdas[pointsL]) + 0.1*np.max(lambdas[pointsL])])
	plt.title("Best lambda(p) for LASSO")
	plt.xlabel("Polynomial degree")
	plt.ylabel(r"$\lambda$")
	plt.savefig(path + "best_lambda_LASSO.pdf")
	plt.show()

	MSE = np.zeros((3, len(polys)))
	R2 = np.zeros((3, len(polys)))
	MSEtrain = np.zeros((3, len(polys)))
	test, train = get_test_train_data(n, test_size=0.20, noise=True)
	xtrain, ytrain, ztrain = train
	xtest, ytest, ztest = test

	for i, p in enumerate(polys):
		ridge = RIDGE(p, lambdas[pointsR[i]])
		lasso = LASSO(p, lambdas[pointsL[i]])
		ols = OLS(p)
		methods = [ols, ridge, lasso]

		Xtrain = ridge.CreateDesignMatrix(xtrain, ytrain)
		Xtest = ridge.CreateDesignMatrix(xtest, ytest)

		for j, method in enumerate(methods):
			method.fit(Xtrain, ztrain)
			MSE[j][i] = method.MSE(ztest, method(Xtest))
			MSEtrain[j][i] = method.MSE(ztrain, method(Xtrain))
			R2[j][i] = method.R2(ztest, method(Xtest))

	labels = ["OLS", "Ridge", "LASSO"]
	for i in range(3):
		plt.plot(polys, MSE[i], label= labels[i])

	plt.xlabel("Polynomial degree")
	plt.ylabel("MSE")
	plt.legend()
	plt.savefig(path + "MSE_with_best_lambdas.pdf")
	plt.show()
	for i in range(3):
		plt.plot(polys, R2[i], label=labels[i])
	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel("R2")
	plt.legend()
	plt.savefig(path + "R2_with_best_lambda.pdf")
	plt.show()

	for i in range(3):
		plt.plot(polys, MSE[i], label= labels[i] + " test")
		plt.plot(polys, MSEtrain[i], label= labels[i] + " training")
		plt.xlabel("Polynomial degree")
		plt.ylabel("MSE")
		plt.legend()
		plt.show()

if False:
	n = 81
	plot_lamb_poly(n, True)


def plot_conf_beta(method, n):
	xn, yn, zn = get_train_data(n, noise=True)
	x, y, z = get_train_data(n, noise=False)
	X = method.CreateDesignMatrix(xn, yn)

	betaSTD_f = method.confIntBeta(X, X, z, z)
	beta_f = method.beta
	betaSTD_z = method.confIntBeta(X, X, zn, zn)
	beta_z = method.beta
	N = len(beta_z)
	colors = ["mediumblue","crimson"]
	plt.plot(-10, -1, color=colors[0], label="without noise")
	plt.plot(-10, -1, color=colors[1], label="with noise")
	plt.legend()

	for i in range(N):
		plt.errorbar(i, beta_f[i], yerr=betaSTD_f[i], capsize=4, \
			color=colors[0], marker='.', markersize=7, elinewidth=2,\
			alpha=0.5)
		plt.errorbar(i, beta_z[i], yerr=betaSTD_z[i], capsize=4, \
			color=colors[1], marker='.', markersize=7, elinewidth=2,\
			alpha=0.5)
	xticks = [r'$\beta_{%d}$'%i for i in range(N)]
	plt.xticks(range(N), xticks, fontsize=fsize)
	plt.xlim(-1, N)
	plt.tight_layout()
	plt.savefig(path + "confIntBeta_" + str(method) + ".pdf")
	#plt.grid()
	plt.show()


if False:
	n = 81
	plot_conf_beta(OLS(5), n)
	plot_conf_beta(RIDGE(5, 0.0001), n)
	plot_conf_beta(LASSO(5, 0.00003), n)


def plot_MSE_test_train(n, method, p_max=20):
	test, train = get_test_train_data(n, 0.2, noise=True)
	p_list = np.arange(1, p_max, 1, dtype=int)
	x_train, y_train, z_train = train
	x_test, y_test, z_test = test
	MSE_train = np.zeros(p_max-1)
	MSE_test = np.zeros(p_max-1)

	for i in range(p_max-1):
		method.p = p_list[i]
		# TRAIN
		X = method.CreateDesignMatrix(x_train, y_train)
		method.fit(X, z_train)
		MSE_train[i] = method.MSE(z_train, method(X))
		# TEST
		X = method.CreateDesignMatrix(x_test, y_test)
		MSE_test[i] = method.MSE(z_test, method(X))
		method.beta = None

	figname = path + "MSE_test_train_" + str(method) + ".pdf"
	plt.plot(p_list, MSE_train, label="Training Sample", color="red")
	plt.plot(p_list, MSE_test, label="Test Sample", color="blue")
	plt.legend(fontsize=fsize)
	plt.xlabel(r"polynomial degree", fontsize=fsize)
	plt.ylabel("Error", fontsize=fsize)
	plt.tight_layout()
	#plt.grid()
	#plt.savefig(figname)
	plt.show()

# plotting MSE for training data & test data
if False:
	n = 81
	plot_MSE_test_train(n, OLS(0), p_max=20)
	#plot_MSE_test_train(n, RIDGE(0, 0.001), p_max=20)
	#plot_MSE_test_train(train_data, test_data, LASSO(0, 0.001), p_max=20)

def bias_variance(n, method, p_max=20):
	test, train = get_test_train_data(n, 0.2, noise=True)
	p_list = np.arange(1, p_max, 1, dtype=int)
	x_train, y_train, z_train = train
	x_test, y_test, z_test = test
	MSE = np.zeros(p_max-1)
	bias = np.zeros(p_max-1)
	variance = np.zeros(p_max-1)
	iterations = 100

	for i, p in enumerate(tqdm.tqdm(p_list)):
		method.p = p
		ztilde = np.empty((len(z_test), iterations))
		for j in range(iterations):
			Xtrain = method.CreateDesignMatrix(x_train, y_train)
			Xtest = method.CreateDesignMatrix(x_test, y_test)
			method.fit(Xtrain, z_train)
			ztilde[:,j] = method(Xtest)
		MSE[i] = np.mean(np.mean((z_test.reshape(-1, 1) - ztilde)**2, axis=1, keepdims=True))
		bias[i] = np.mean( (z_test.reshape(-1, 1) - np.mean(ztilde, axis=1, keepdims=True))**2 )
		variance[i] = np.mean( np.var(ztilde, axis=1, keepdims=True) )

	plt.plot(p_list, MSE, label="MSE")
	plt.plot(p_list, bias, label="Bias")
	plt.plot(p_list, variance, label="Variance")
	plt.legend()
	plt.show()

if False:
	n = 81
	bias_variance(n, OLS(0), 8)
