from franke import *
from regClass import OLS, LASSO, RIDGE
import matplotlib.pyplot as plt 
import numpy as np

# I AM SO SORRY
import warnings
warnings.filterwarnings('ignore')


fsize = 15 				# universal fontsize for plots
path = "figures/"



def plot_MSE_R2(n, noise):
	p = 5
	ols = OLS(p)
	lasso = LASSO(p, 0)
	ridge = RIDGE(p, 0)

	M = 50
	lambdas = np.linspace(-4, -1, M) # log_10
	MSE = np.zeros((3, M))
	R2 = np.zeros((3, M))
	for i in range(M):
		lasso.l = 10**lambdas[i]
		ridge.l = 10**lambdas[i]
		# TRAIN
		x, y, z = get_train_data(n, noise=noise)
		X = ols.CreateDesignMatrix(x, y)
		ols.fit(X, z)
		lasso.fit(X, z)
		ridge.fit(X, z)

		# ---- delete if test is on train data
		x, y, z = get_test_data(50, noise=noise)
		X = ols.CreateDesignMatrix(x, y)
		# ...............................

		MSE[0][i] = ols.MSE(z, ols(X))
		MSE[1][i] = ridge.MSE(z, ridge(X))
		MSE[2][i] = lasso.MSE(z, lasso(X))

		R2[0][i] = ols.R2(z, ols(X))
		R2[1][i] = ridge.R2(z, ridge(X))
		R2[2][i] = lasso.R2(z, lasso(X))


	title = " for n = %d " %n
	if noise:
		end = str(n) + "_noise.pdf"
		title += "with noise"
	else:
		end = str(n) + ".pdf"
		title += "without noise"
	plt.plot(lambdas, MSE[0], label="OLS")
	plt.plot(lambdas, MSE[1], label="RIDGE")
	plt.plot(lambdas, MSE[2], label="LASSO")
	plt.title("MSE" + title)
	plt.legend(fontsize=fsize)
	plt.xlabel(r"$log_{10}(\lambda)$", fontsize=fsize)
	plt.ylabel("MSE", fontsize=fsize)
	plt.tight_layout()
	plt.savefig(path + "MSE_methods_" + end)
	plt.show()

	plt.plot(lambdas, R2[0], label="OLS")
	plt.plot(lambdas, R2[1], label="RIDGE")
	plt.plot(lambdas, R2[2], label="LASSO")
	plt.title(r"$R^2$" + title)
	plt.legend(fontsize=fsize)
	plt.xlabel(r"$log_{10}(\lambda)$", fontsize=fsize)
	plt.ylabel("$R^2$", fontsize=fsize)
	plt.tight_layout()
	plt.savefig(path + "R2_methods_" + end)
	plt.show()


if True:
	n = 20
	plot_MSE_R2(n, True)
	plot_MSE_R2(n, False)

	n = 50
	plot_MSE_R2(n, True)
	plot_MSE_R2(n, False)

	n = 100
	plot_MSE_R2(n, True)
	plot_MSE_R2(n, False)




def plot_MSE_test_train(train_data, test_data, method, p_max=20):
	p_list = np.arange(1, p_max, 1, dtype=int)
	x_train, y_train, z_train = train_data
	x_test, y_test, z_test = test_data
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


	figname = path + "MSE_test_train_" + str(method) + ".pdf"
	plt.plot(p_list, MSE_train, label="Training Sample", color="red")
	plt.plot(p_list, MSE_test, label="Test Sample", color="blue")
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
	plot_MSE_test_train(train_data, test_data, RIDGE(0 ,0.001), p_max=20)
	plot_MSE_test_train(train_data, test_data, LASSO(0, 0.001), p_max=20)



