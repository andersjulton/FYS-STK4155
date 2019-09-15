from regClass import OLS, LASSO, RIDGE
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from franke import FrankeFunction, get_train_data, get_test_data
import warnings
warnings.filterwarnings('ignore')

compute = True
p = 4
show = False


n_list = [i*2 for i in range(10, 101)]
N = len(n_list)
M = 100
lambdas = np.linspace(1e-4, 0.1, M)

path = "store_results/p%d_" %p

if compute:
	x_test, y_test, z_test = get_test_data(100)

	OLS_RR = np.zeros(N)
	RIDGE_RR = np.zeros((N, M))
	LASSO_RR = np.zeros((N, M))

	OLS_MSE_train = np.zeros(N)
	RIDGE_MSE_train = np.zeros((N, M))
	LASSO_MSE_train = np.zeros((N, M))

	OLS_MSE_test = np.zeros(N)
	RIDGE_MSE_test = np.zeros((N, M))
	LASSO_MSE_test = np.zeros((N, M))

	print("|"+ "_"*N +"|")
	print("|", end="", flush=True)
	for i in range(N):
		n = n_list[i]
		print("*", end="", flush=True)
		x, y, z = get_train_data(n)
		franke_OLS = OLS(x, y, z, p)
		OLS_RR[i] = franke_OLS.RR()
		OLS_MSE_train[i] = franke_OLS.MSE()
		z_tilde = franke_OLS(x_test, y_test)
		OLS_MSE_test[i] = franke_OLS.MSE(z=z_test, z_tilde=z_tilde)
		for j in range(M):
			l = lambdas[j]
			franke_RIDGE = RIDGE(x, y, z, p, l)
			RIDGE_RR[i][j] = franke_RIDGE.RR()
			RIDGE_MSE_train[i][j] = franke_RIDGE.MSE()
			z_tilde = franke_RIDGE(x_test, y_test)
			RIDGE_MSE_test[i][j] = franke_RIDGE.MSE(z=z_test, z_tilde=z_tilde)

			franke_LASSO = LASSO(x, y, z, p, l)
			LASSO_RR[i][j] = franke_LASSO.RR()
			LASSO_MSE_train[i][j] = franke_LASSO.MSE()
			z_tilde = franke_LASSO(x_test, y_test)
			LASSO_MSE_test[i][j] = franke_LASSO.MSE(z=z_test, z_tilde=z_tilde)

	RR = {"OLS" : OLS_RR, "LASSO" : LASSO_RR, "RIDGE" : RIDGE_RR}
	np.savez(path+"RR", **RR)

	TRAIN = {"OLS" : OLS_MSE_train, "LASSO" : LASSO_MSE_train, "RIDGE" : RIDGE_MSE_train}
	TEST = {"OLS" : OLS_MSE_test, "LASSO" : LASSO_MSE_test, "RIDGE" : RIDGE_MSE_test}
	np.savez(path+"MSE_TRAIN", **TRAIN)
	np.savez(path+"MSE_TEST", **TEST)

	print("|")
else:
	RR = np.load(path+"RR.npz"%p)
	TRAIN = np.load(path+"MSE_TRAIN.npz")
	TEST = np.load(path+"MSE_TEST.npz")


path = "figures/p%d_" %p
RR_min = 0.6; RR_max = 1
MSE_min = 0; MSE_max = 0.1 
# plot OLS, 2D
plt.figure(figsize=(12,10))
plt.plot(n_list, RR["OLS"])
plt.title(r"OLS $R^2$")
plt.xlabel(r"n")
plt.ylabel(r"$R^2$")
#plt.ylim(RR_min, RR_max)
plt.savefig(path + "RR_OLS.png")
if show:
	plt.show()
else:
	plt.close()

plt.figure(figsize=(12,10))
plt.plot(n_list, TRAIN['OLS'])
plt.title(r"OLS $MSE$ on train data")
plt.xlabel(r"n")
plt.ylabel(r"MSE")
#plt.ylim(MSE_min, MSE_max)
plt.savefig(path + "MSE_train_OLS.png")
if show:
	plt.show()
else:
	plt.close()

plt.figure(figsize=(12,10))
plt.plot(n_list, TEST["OLS"])
plt.title(r"OLS $MSE$ on test data")
plt.xlabel(r"n")
plt.ylabel(r"MSE")
#plt.ylim(MSE_min, MSE_max)
plt.savefig(path + "MSE_test_RR_OLS.png")
if show:
	plt.show()
else:
	plt.close()


# plot RIDGE and LASSO, 3D
cmap_RR = plt.get_cmap("inferno")
cmap_MSE = plt.get_cmap("summer")
Y, X = np.meshgrid(lambdas, n_list)
print(X.shape, Y.shape)

for name in ["RIDGE", "LASSO"]:
	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(1, 1, 1)
	plt.title(r"%s $R^2$" %name)
	im = ax.pcolormesh(X, Y, RR[name], cmap=cmap_RR)#, vmin=RR_min, vmax=RR_max)
	plt.xlabel(r"n")
	plt.ylabel(r"$\lambda$")
	fig.colorbar(im, ax=ax)
	plt.savefig(path + "RR_%s.png" %name)
	if show:
		plt.show()
	else:
		plt.close()

	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(1, 1, 1)
	plt.title(r"%s MSE on train data" %name)
	im = ax.pcolormesh(X, Y, TRAIN[name], cmap=cmap_MSE)#, vmin=MSE_min, vmax=MSE_max)
	plt.xlabel(r"n")
	plt.ylabel(r"$\lambda$")
	fig.colorbar(im, ax=ax)
	plt.savefig(path + "MSE_train_%s.png" %name)
	if show:
		plt.show()
	else:
		plt.close()

	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(1, 1, 1)
	plt.title(r"%s MSE on test data" %name)
	im = ax.pcolormesh(X, Y, TEST[name], cmap=cmap_MSE)#, vmin=MSE_min, vmax=MSE_max)
	plt.xlabel(r"n")
	plt.ylabel(r"$\lambda$")
	fig.colorbar(im, ax=ax)
	plt.savefig(path + "MSE_test_%s.png" %name)
	if show:
		plt.show()
	else:
		plt.close()





