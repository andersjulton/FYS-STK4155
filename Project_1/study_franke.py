from franke import *
from regClass import OLS, LASSO, RIDGE
import matplotlib.pyplot as plt
import numpy as np
import tqdm

#np.random.seed(2019)

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
	N = np.linspace(50, 150, 101, dtype="int16")
	m = np.linspace(0, 9, 10, dtype="int16")
	MSE, R2 = np.zeros((10,101)), np.zeros((10, 101))

	for i in tqdm.tqdm(m):
		for j, n in enumerate(N):
			x, y, z = get_train_data(n, noise=False)
			xn, yn, zn = get_train_data(n, noise=True)
			Xn = ols.CreateDesignMatrix(xn, yn)
			X = ols.CreateDesignMatrix(x, y)
			ols.fit(Xn, zn)

			MSE[i][j] = ols.MSE(z, ols(X))
			R2[i][j] = ols.R2(z, ols(X))

	plt.plot(N, np.mean(MSE, axis=0))
	plt.ylabel("MSE", fontsize=fsize)
	plt.xlabel("n", fontsize=fsize)
	plt.savefig(path + "OLS(n)_MSE.pdf")
	plt.show()

	plt.plot(N, np.mean(R2, axis=0))
	plt.ylabel("R2", fontsize=fsize)
	plt.xlabel("n", fontsize=fsize)
	plt.savefig(path + "OLS(n)_R2.pdf")

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

	M = 200
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
		plt.savefig(path + "MSE_methods_" + labels[i] + "_" + end)
		plt.show()

		plt.semilogx(lambdas[i], R2[i], label=labels[i])
		plt.scatter(lambdas[i][maxR2], R2[i][maxR2], color='r', label=r"$\lambda = %1.5f$" % lambdas[i][maxR2])
		plt.title(r"$R^2$" + title)
		plt.legend(fontsize=fsize)
		plt.xlabel(r"$log_{10}(\lambda)$", fontsize=fsize)
		plt.ylabel("$R^2$", fontsize=fsize)
		plt.tight_layout()
		plt.savefig(path + "R2_methods_" + labels[i] + "_" + end)
		plt.show()


# plotting RIDGE and LASSO as functions of lambda
if False:
	n = 50
	plot_MSE_R2(n, True)


def plot_conf_beta(method, n):
	x, y, z = get_train_data(n, noise=False)
	betaSTD_f = method.confIntBeta(x, y, z)
	beta_f = method.beta
	x, y, z = get_train_data(n, noise=True)
	betaSTD_z = method.confIntBeta(x, y, z)
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
	plt.grid()
	plt.show()


if False:
	n = 100
	plot_conf_beta(OLS(5), n)
	plot_conf_beta(RIDGE(5, 0.001), n)
	plot_conf_beta(LASSO(5, 0.0005), n)


def plot_MSE_test_train(train_data, test_data, method, p_max=20):
	p_list = np.arange(1, p_max, 1, dtype=int)
	x_train, y_train, z_train = train_data
	x_test, y_test, z_test = test_data
	MSE_train = np.zeros(p_max-1)
	MSE_test = np.zeros(p_max-1)
	bias = np.zeros(p_max-1)
	variance = np.zeros(p_max-1)

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
	plt.plot(p_list, bias, label="bias", color="green")
	plt.plot(p_list, variance, label="variance", color="yellow")
	plt.legend(fontsize=fsize)
	plt.xlabel(r"polynomial degree", fontsize=fsize)
	plt.ylabel("MSE", fontsize=fsize)
	plt.tight_layout()
	#plt.grid()
	plt.savefig(figname)
	plt.show()


# plotting MSE for training data & test data
if False:
	n = 50
	train_data = get_train_data(n, noise=True)
	test_data = get_test_data(n, noise=True)
	plot_MSE_test_train(train_data, test_data, OLS(0), p_max=10)
	#plot_MSE_test_train(train_data, test_data, RIDGE(0 ,0.001), p_max=20)
	#plot_MSE_test_train(train_data, test_data, LASSO(0, 0.001), p_max=20)
