from franke import *
from regClass import OLS, LASSO, RIDGE
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import tqdm
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings('ignore')

compute_OLS_n = False
compute_conf_beta = True
compute_best_lambdas = False
compute_bias_variance = False
compute_MSE_R2_lambdas = False
compute_kfold = False
compute_MSE_test_train = False
plot_Franke = False
compute_dif_STD = False

fsize = 13				# universal fontsize for plots
path = "figures/"

if compute_OLS_n:
	"""
	OLS as function of n, mean over 10 runs.
	Comparing score with test and train data.
	"""
	K = 10
	ols = OLS(5)
	N = np.arange(10, 100, 1, dtype="int")
	M = len(N)
	MSE_train, R2_train = np.zeros(M), np.zeros(M)
	MSE_test, R2_test = np.zeros(M), np.zeros(M)

	for i in tqdm.tqdm(range(M)):
		n = N[i]
		for j in range(K):
			# TRAIN DATA
			x, y, z = get_train_data(n, noise=True)
			X = ols.CreateDesignMatrix(x, y)
			ols.fit(X, z)

			z_tilde = ols(X)
			MSE_train[i] += ols.MSE(z, z_tilde)
			R2_train[i] += ols.R2(z, z_tilde)

			# TEST DATA
			x, y, z = get_train_data(n, noise=True)
			X = ols.CreateDesignMatrix(x, y)

			z_tilde = ols(X)
			MSE_test[i] += ols.MSE(z, z_tilde)
			R2_test[i] += ols.R2(z, z_tilde)

	plt.plot(N, MSE_train/K, label="Train data")
	plt.plot(N, MSE_test/K, label="Test data")
	plt.ylabel("MSE", fontsize=fsize)
	plt.xlabel(r"n ($n^2$ number of data points)", fontsize=fsize)
	plt.legend(fontsize=fsize)
	plt.ylim(0.8, 1.2)
	plt.tight_layout()
	plt.savefig(path + "OLS(n)_MSE.pdf")
	plt.show()

	plt.plot(N, R2_train/K, label="Train data")
	plt.plot(N, R2_test/K, label="Test data")
	plt.ylabel(r"$R^2$", fontsize=fsize)
	plt.xlabel(r"n ($n^2$ number of data points)", fontsize=fsize)
	plt.legend(fontsize=fsize)
	plt.ylim(0, 0.25)
	plt.tight_layout()
	plt.savefig(path + "OLS(n)_R2.pdf")
	plt.show()


if compute_MSE_R2_lambdas:
	np.random.seed(10101)
	n = 81
	noise = True
	p = 5
	"""
	Ridge and LASSO as function of lambda.
	"""
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
	colors = ["mediumblue","crimson"]

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
		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		ax1.semilogx(lambdas[i], MSE[i], label="MSE", color=colors[0])
		ax2.semilogx(lambdas[i], R2[i], label=r"$R^2$", color=colors[1])
		ax1.legend(loc=6, fontsize=fsize)
		ax2.legend(loc=10, fontsize=fsize)
		ax2.tick_params(axis='y', color=colors[1])
		plt.xlabel(r"$log_{10}(\lambda)$", fontsize=fsize)
		ax1.set_ylabel("MSE", fontsize=fsize)
		ax2.set_ylabel(r"$R^2$", fontsize=fsize, color=colors[1])

		plt.tight_layout()
		plt.savefig(path + "MSE_methods_" + labels[i] + "_" + end)
		plt.show()


def plot_kfold(n, method, noise, k=10):

	test, train = get_test_train_data(n, 0.2, noise)
	xtest, ytest, ztest = test
	xtrain, ytrain, ztrain = train

	R2, MSE_, MSE = method.kFoldCV(xtrain, ytrain, ztrain,k)
	print(f"Mean MSE = {MSE_}")
	ind = np.arange(1, k+1, 1)
	plt.bar(ind, MSE)
	plt.plot([0.5, k+0.5], np.zeros(2) + MSE_, "--", color="crimson") # mean
	plt.xticks(ind)
	plt.ylabel("MSE", fontsize=fsize)
	plt.xlabel("Fold", fontsize=fsize)
	plt.tight_layout()
	plt.savefig(path + "kfold.pdf")
	plt.show()

if compute_kfold:
	np.random.seed(42)
	n = 50
	plot_kfold(n, OLS(5), True)

if compute_best_lambdas:
	np.random.seed(42)
	noise = False
	n = 125
	M = 80
	lambdas = np.logspace(-7, -1, M)

	polys = np.arange(4, 20)
	N = len(polys)

	MSE_ols = np.zeros(N)
	R2_ols = np.zeros(N)

	MSE_ridge = np.zeros(N)
	R2_ridge = np.zeros(N)
	lambda_ridge = np.zeros(N) + lambdas[0]

	MSE_lasso = np.zeros(N)
	R2_lasso = np.zeros(N)
	lambda_lasso = np.zeros(N) + lambdas[0]

	test, train = get_test_train_data(n, test_size=0.2, noise=noise)
	xtrain, ytrain, ztrain = train
	xtest, ytest, ztest = test


	for i, p in enumerate(tqdm.tqdm(polys)):
		ridge = RIDGE(p, lambdas[0])
		lasso = LASSO(p, lambdas[0])
		ols = OLS(p)

		Xtrain = ridge.CreateDesignMatrix(xtrain, ytrain)
		Xtest = ridge.CreateDesignMatrix(xtest, ytest)

		ridge.fit(Xtrain, ztrain)
		z_ridge = ridge(Xtest)
		MSE_ridge[i] = ridge.MSE(ztest, z_ridge)
		R2_ridge[i] = ridge.R2(ztest, z_ridge)

		lasso.fit(Xtrain, ztrain)
		z_lasso = lasso(Xtest)
		MSE_lasso[i] = lasso.MSE(ztest, z_lasso)
		R2_lasso[i] = lasso.R2(ztest, z_lasso)

		ols.fit(Xtrain, ztrain)
		z_ols = ols(Xtest)
		MSE_ols[i] = lasso.MSE(ztest, z_ols)
		R2_ols[i] = lasso.R2(ztest, z_ols)

		for j in range(1, M):
			ridge.l = lambdas[j]
			ridge.fit(Xtrain, ztrain)
			z_ridge = ridge(Xtest)
			MSE = ridge.MSE(ztest, z_ridge)
			if MSE <= MSE_ridge[i]:
				MSE_ridge[i] = MSE
				R2_ridge[i] = ridge.R2(ztest, z_ridge)
				lambda_ridge[i] = lambdas[j]

			lasso.l = lambdas[j]
			lasso.fit(Xtrain, ztrain)
			z_lasso = lasso(Xtest)
			MSE = lasso.MSE(ztest, z_lasso)
			if MSE <= MSE_lasso[i]:
				MSE_lasso[i] = MSE
				R2_lasso[i] = ridge.R2(ztest, z_lasso)
				lambda_lasso[i] = lambdas[j]

	# plot best lambdas
	def y_ticks(l):
		l =  np.log10(l)
		l_min = int(min(l))
		l_max = int(max(l)) + 1
		return range(l_min, l_max), [r"$10^{%d}$" %i for i in range(l_min, l_max)]


	plt.plot(polys, np.log10(lambda_ridge), '--')
	plt.scatter(polys, np.log10(lambda_ridge))
	plt.yticks(*y_ticks(lambda_ridge))
	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel(r"$\lambda$", fontsize=fsize)
	plt.tight_layout()
	plt.savefig("best_lambda_RIDGE.pdf")
	plt.show()

	plt.plot(polys, np.log10(lambda_lasso), '--')
	plt.scatter(polys, np.log10(lambda_lasso))
	plt.yticks(*y_ticks(lambda_lasso))
	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel(r"$\lambda$", fontsize=fsize)
	plt.tight_layout()
	plt.savefig("best_lambda_LASSO.pdf")
	plt.show()


	# plot MSE for best lambdas
	plt.plot(polys, MSE_ols, label="OLS")
	plt.plot(polys, MSE_ridge, label="Ridge")
	plt.plot(polys, MSE_lasso, label="LASSO")
	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel("MSE", fontsize=fsize)
	#plt.ylim(1, 1.1)
	plt.legend(fontsize=fsize)
	plt.tight_layout()
	plt.savefig("MSE_with_best_lambdas.pdf")
	plt.show()


	# plot R2 for best lambdas
	point = np.argmax(R2_ridge)
	plt.plot(polys, R2_ols, label="OLS")
	plt.plot(polys, R2_ridge, label="Ridge")
	plt.plot(polys, R2_lasso, label="LASSO")
	plt.scatter(polys[point], R2_ridge[point], color='r', label = r"p = {}, $R^2$ = {:2.3f}".format(polys[point], R2_ridge[point]))
	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel(r"$R^2$", fontsize=fsize)
	plt.legend(fontsize=fsize)
	#plt.ylim(0.06, 0.09)
	plt.tight_layout()
	plt.savefig("R2_with_best_lambda.pdf")
	plt.show()

def get_best_lambda(n, method, lmin=-9, lmax=-2, M=100):
	lambdas = np.logspace(lmin, lmax, M)
	test, train = get_test_train_data(n, 0.2, True)
	xtrain, ytrain, ztrain = train
	xtest, ytest, ztest = test
	Xtrain = method.CreateDesignMatrix(xtrain, ytrain)
	Xtest = method.CreateDesignMatrix(xtest, ytest)
	method.l = lambdas[0]
	method.fit(Xtrain, ztrain)
	ztilde = method(Xtest)
	MSEprev = method.MSE(ztest, ztilde)
	lamb = lambdas[0]

	for i in range(1, M):
		method.l = lambdas[i]
		method.fit(Xtrain, ztrain)
		ztilde = method(Xtest)
		MSE = method.MSE(ztest, ztilde)
		if MSE <= MSEprev:
			MSEprev = MSE
			lamb = lambdas[i]
	return lamb


def plot_conf_beta(method, n, alpha):
	#if str(method) == "RIDGE":
		#method.l = get_best_lambda(n, method)
	xn, yn, zn = get_train_data(n, noise=True)
	x, y, z = get_train_data(n, noise=False)
	Xn = method.CreateDesignMatrix(xn, yn)
	X = method.CreateDesignMatrix(x, y)

	betaSTD_f, zSTDf = method.confIntBeta(X, X, z, z, alpha)
	beta_f = method.beta
	betaSTD_z, zSTDz = method.confIntBeta(Xn, Xn, zn, zn, alpha)
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
	plt.xticks(range(N), xticks, fontsize=fsize, rotation=-45)
	plt.xlim(-1, N)
	plt.tight_layout()
	plt.savefig("confIntBeta_" + str(method) + ".pdf")
	plt.show()

def plot_dif_STD(n, alpha):
	ols = OLS(5)
	l = get_best_lambda(n, RIDGE(5, 0))
	l = 7.4e-4
	ridge = RIDGE(5, l)

	xn, yn, zn = get_train_data(n, noise=True)
	Xn = ols.CreateDesignMatrix(xn, yn)

	betaSTDols, olsSTD = ols.confIntBeta(Xn, Xn, zn, zn, alpha)
	betaSTDridge, ridgeSTD = ridge.confIntBeta(Xn, Xn, zn, zn, alpha)

	beta_z = ols.beta
	N = len(beta_z)
	arr = np.arange(0, N)

	plt.scatter(arr, olsSTD, color='green', label="OLS")
	plt.plot(olsSTD, linestyle='--', color='green')
	plt.scatter(arr, ridgeSTD, color='blue', label="Ridge")
	plt.plot(ridgeSTD, linestyle='--', color='blue')
	plt.legend(loc='best', fontsize=fsize)
	xticks = [r'$\beta_{%d}$'%i for i in range(N)]
	plt.xticks(range(N), xticks, fontsize=fsize, rotation=-45)
	plt.ylabel(r"$\sqrt{Var(\beta)}$", fontsize=fsize)
	plt.savefig("STD_difference.pdf")
	plt.show()

if compute_conf_beta:
	np.random.seed(42)
	n = 80
	alpha = 1.96
	olsSTD = plot_conf_beta(OLS(5), n, alpha)
	ridgeSTD = plot_conf_beta(RIDGE(5, 7.4e-4), n, alpha)

if compute_dif_STD:
	#np.random.seed(42)
	n = 80
	alpha = 1.96
	plot_dif_STD(n, alpha)

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
	plt.ylabel("MSE", fontsize=fsize)
	plt.xticks(p_list)
	plt.tight_layout()
	plt.savefig(figname)
	plt.show()

# plotting MSE for training data & test data
if compute_MSE_test_train:
	np.random.seed(42)
	n = 81
	plot_MSE_test_train(n, OLS(0), p_max=20)
	#plot_MSE_test_train(n, RIDGE(0, 0.001), p_max=20)
	#plot_MSE_test_train(train_data, test_data, LASSO(0, 0.001), p_max=20)

def bias_variance(n, method, p_max=20):
	p_list = np.arange(1, p_max, 1, dtype=int)

	bias = np.zeros(len(p_list))
	variance = np.zeros(len(p_list))
	iterations = 100
	test, train = get_test_train_data(n, 0.2, noise=False)
	x_train, y_train, z_train = train
	x_test, y_test, z_test = test
	z_test1 = z_test

	for i, p in enumerate(tqdm.tqdm(p_list)):
		method.p = p
		ztilde = np.empty((len(z_test), iterations))
		for j in range(iterations):
			test, train = get_test_train_data(n, 0.2, noise=True)
			x_train, y_train, z_train = train
			x_test, y_test, z_test = test
			Xtrain = method.CreateDesignMatrix(x_train, y_train)
			Xtest = method.CreateDesignMatrix(x_test, y_test)
			method.fit(Xtrain, z_train)
			ztilde[:,j] = method(Xtest)
		bias[i] = np.mean( (z_test1.reshape(-1, 1) - np.mean(ztilde, axis=1, keepdims=True))**2 )
		variance[i] = np.mean( np.var(ztilde, axis=1, keepdims=True) )

	return p_list, bias, variance


if compute_bias_variance:
	np.random.seed(666)
	n = 81
	colors = ['crimson', 'mediumblue', 'orange']

	p, bias, var = bias_variance(n, OLS(0), 14)
	plt.plot(p, bias, linestyle='dashed', color=colors[1], label=r"bias$^2$")
	plt.plot(p, var, color=colors[1], label=r"variance")
	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel("Error", fontsize=fsize)
	plt.legend(loc='best', fontsize=fsize)
	plt.tight_layout()
	plt.savefig(path + "bias_variance_OLS.pdf")
	plt.show()

	lambdas = [5e-4, 5e-5, 5e-6]
	lambda_label = [r"5 \cdot 10^{-4}", r"5 \cdot 10^{-5}", r"5 \cdot 10^{-6}"]
	for i in range(3):
		p, bias, var = bias_variance(n, LASSO(0, lambdas[i]), 14)
		plt.plot(p, bias, linestyle='dashed', color=colors[i])
		plt.plot(p, var, label=r"$\lambda = {}$".format(lambda_label[i]), color=colors[i])

	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel("Error", fontsize=fsize)
	plt.legend(loc='best', fontsize=fsize)
	plt.tight_layout()
	plt.savefig(path + "bias_variance_LASSO_lambdas.pdf")
	plt.show()


	lambdas = [0.5, 0.05, 0.005]
	lambda_label = [r"5 \cdot 10^{-1}", r"5 \cdot 10^{-2}", r"5 \cdot 10^{-3}"]
	colors = ['crimson', 'mediumblue', 'orange']
	for i in range(3):
		p, bias, var = bias_variance(n, RIDGE(0, lambdas[i]), 14)
		plt.plot(p, bias, linestyle='dashed', color=colors[i])
		plt.plot(p, var, label=r"$\lambda = {}$".format(lambda_label[i]), color=colors[i])
	plt.xlabel("Polynomial degree", fontsize=fsize)
	plt.ylabel("Error", fontsize=fsize)
	plt.legend(loc='best', fontsize=fsize)
	plt.tight_layout()
	plt.savefig(path + "bias_variance_Ridge_lambdas.pdf")
	plt.show()

if plot_Franke:
	p = 15
	n = 81
	ax = plot_frankeFunc(n)
	ax.view_init(30, 60)
	plt.xlabel(r"x", fontsize=fsize)
	plt.ylabel(r"y", fontsize=fsize)
	ax.set_zlabel(r"$f(x, y)$", fontsize=fsize)
	plt.savefig("figures/franke.pdf")
	plt.show()

	"""lasso = LASSO(p, get_best_lambda(n, LASSO(p, 0)))
	x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
	x, y = np.ravel(x), np.ravel(y)
	z = lasso(x, y)
	plt.scatter(x, y, z, c=z, cmap="coolwarm")
	plt.show()
"""
