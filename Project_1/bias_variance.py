from regClass import OLS, LASSO, RIDGE
import matplotlib.pyplot as plt 
import numpy as np
from franke import FrankeFunction, get_train_data, get_test_data

compute = True
p_list = np.arange(1, 20, 1, dtype=int)
n = 50

N = len(p_list)
path = "store_results/bias_variance_low_n"
#path = "store_results/bias_variance_test"



if compute:
	OLS_MSE_train = np.zeros(N)
	RIDGE_MSE_train = np.zeros(N)
	LASSO_MSE_train = np.zeros(N)

	OLS_MSE_test = np.zeros(N)
	RIDGE_MSE_test = np.zeros(N)
	LASSO_MSE_test = np.zeros(N)


	print("|"+ "_"*N +"|")
	print("|", end="", flush=True)

	# create data
	x_train, y_train, f = get_train_data(n)
	z_train = f + np.random.uniform(low=-1, high=1, size=len(f))

	x_test, y_test, f  = get_test_data(n)
	z_test = f + np.random.uniform(low=-1, high=1, size=len(f))
	for i in range(N):
		print("*", end="", flush=True)
		p = p_list[i]
		ols = OLS(p)
		lasso = LASSO(p, 0.001)
		ridge = RIDGE(p, 0.001)

		# TRAIN
		X = ols.CreateDesignMatrix(x_train, y_train)

		ols.fit(X, z_train)
		lasso.fit(X, z_train)
		ridge.fit(X, z_train)

		OLS_MSE_train[i] = ols.MSE(z_train, ols(X))
		LASSO_MSE_train[i] = lasso.MSE(z_train, lasso(X))
		RIDGE_MSE_train[i] = ridge.MSE(z_train, ridge(X))

		# TEST
		X = ols.CreateDesignMatrix(x_test, y_test)

		OLS_MSE_test[i] = ols.MSE(z_test, ols(X))
		LASSO_MSE_test[i] = lasso.MSE(z_test, lasso(X))
		RIDGE_MSE_test[i] = ridge.MSE(z_test, ridge(X))
	print("|")

	OLS_MSE = {"TRAIN" : OLS_MSE_train,  "TEST" : OLS_MSE_test}
	LASSO_MSE = {"TRAIN" : LASSO_MSE_train,  "TEST" : LASSO_MSE_test}
	RIDGE_MSE = {"TRAIN" : RIDGE_MSE_train,  "TEST" : RIDGE_MSE_test}
	np.savez(path+"OLS", **OLS_MSE)
	np.savez(path+"RIDGE", **RIDGE_MSE)
	np.savez(path+"LASSO", **LASSO_MSE)
else:
	OLS_MSE = np.load(path+"OLS.npz")
	LASSO_MSE = np.load(path+"LASSO.npz")
	RIDGE_MSE = np.load(path+"RIDGE.npz")

def plot_bias_variance(p, MSE, method):
	s = 20
	figname = "figures/" + "bias_variance_" + method + ".pdf"
	plt.plot(p, MSE["TRAIN"], label="Training Sample", color="red")
	plt.plot(p, MSE["TEST"], label="Test Sample", color="blue")
	plt.legend(fontsize=s)
	plt.xlabel(r"polynomial degree", fontsize=s)
	plt.ylabel("MSE", fontsize=s)
	plt.grid()
	plt.savefig(figname)
	plt.show()


plot_bias_variance(p_list, OLS_MSE, "OLS")
plot_bias_variance(p_list, RIDGE_MSE, "RIDGE")
plot_bias_variance(p_list, LASSO_MSE, "LASSO")

cut = 9
p_list = p_list[0: cut]

OLS_MSE_cut = {}

OLS_MSE_cut["TRAIN"] = OLS_MSE["TRAIN"][0:cut]
OLS_MSE_cut["TEST"] = OLS_MSE["TEST"][0:cut]

plot_bias_variance(p_list, OLS_MSE_cut, "OLS_cut")



