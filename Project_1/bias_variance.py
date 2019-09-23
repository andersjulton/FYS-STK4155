from regClass import OLS, LASSO, RIDGE
import matplotlib.pyplot as plt 
import numpy as np
from franke import FrankeFunction, get_train_data, get_test_data


n_list = np.arange(30, 500, 5, dtype=int)

N = len(n_list)
OLS_MSE_train = np.zeros(N)
RIDGE_MSE_train = np.zeros(N)
LASSO_MSE_train = np.zeros(N)

OLS_MSE_test = np.zeros(N)
RIDGE_MSE_test = np.zeros(N)
LASSO_MSE_test = np.zeros(N)


p = 5
ols = OLS(p)
lasso = LASSO(p, 0.001)
ridge = RIDGE(p, 0.001)

print("|"+ "_"*N +"|")
print("|", end="", flush=True)
for i in range(N):
	print("*", end="", flush=True)
	x, y, f = get_train_data(n_list[i])
	X = ols.CreateDesignMatrix(x, y)
	z = f + np.random.randn(len(f))

	ols.fit(X, z)
	lasso.fit(X, z)
	ridge.fit(X, z)

	OLS_MSE_train[i] = ols.MSE(z, ols(X))
	LASSO_MSE_train[i] = lasso.MSE(z, lasso(X))
	RIDGE_MSE_train[i] = ridge.MSE(z, ridge(X))

	x, y, f  = get_test_data(n_list[i])
	X = ols.CreateDesignMatrix(x, y)
	z = f + np.random.randn(len(f))

	OLS_MSE_test[i] = ols.MSE(z, ols(X))
	LASSO_MSE_test[i] = lasso.MSE(z, lasso(X))
	RIDGE_MSE_test[i] = ridge.MSE(z, ridge(X))
print("|")

def plot_bias_variance(n, MSE_train, MSE_test, method):
	figname = "figures/" + method + "_bias_variance.pdf"
	plt.plot(n, MSE_train, label="Training Sample", color="red")
	plt.plot(n, MSE_test, label="Test Sample", color="blue")
	plt.legend()
	plt.savefig(figname)
	plt.xlabel(r"$n$ number of points")
	plt.ylabel("MSE")
	plt.show()


plot_bias_variance(n_list, OLS_MSE_train, OLS_MSE_test, "OLS")
plot_bias_variance(n_list, RIDGE_MSE_train, RIDGE_MSE_test, "RIDGE")
plot_bias_variance(n_list, LASSO_MSE_train, LASSO_MSE_test, "LASSO")



