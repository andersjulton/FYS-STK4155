from regClass import OLS, LASSO, RIDGE
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from franke import FrankeFunction, get_train_data, get_test_data
import warnings
warnings.filterwarnings('ignore')

compute = False
show = False
noise = True


p = 5
n_list = np.arange(50, 500, 10, dtype=int)
N = len(n_list)
M = 100
lambdas = np.linspace(1e-4, 0.1, M)

path = "store_results/"

if noise:
	path += "noise_"

if compute:
	OLS_RR = np.zeros(N)
	RIDGE_RR = np.zeros((N, M))
	LASSO_RR = np.zeros((N, M))

	OLS_MSE = np.zeros(N)
	RIDGE_MSE = np.zeros((N, M))
	LASSO_MSE = np.zeros((N, M))

	print("|"+ "_"*N +"|")
	print("|", end="", flush=True)
	for i in range(N):
		print("*", end="", flush=True)

		x, y, f = get_train_data(n_list[i])

		if noise:
			epsilon = np.random.randn(len(f))
			z = f + epsilon
		else:
			z = f

		franke_OLS = OLS(p)
		X = franke_OLS.CreateDesignMatrix(x, y)
		franke_OLS.fit(X, z)
		z_tilde = franke_OLS(X)

		OLS_RR[i] = franke_OLS.R2(z, z_tilde)
		OLS_MSE[i] = franke_OLS.MSE(z, z_tilde)
		for j in range(M):
			l = lambdas[j]

			franke_RIDGE = RIDGE(p, l)
			franke_RIDGE.fit(X, z)
			z_tilde = franke_RIDGE(X)

			RIDGE_RR[i][j] = franke_RIDGE.R2(z, z_tilde)
			RIDGE_MSE[i][j] = franke_RIDGE.MSE(z, z_tilde)


			franke_LASSO = LASSO(p, l)
			franke_LASSO.fit(X, z)
			z_tilde = franke_LASSO(X)

			LASSO_RR[i][j] = franke_LASSO.R2(z, z_tilde)
			LASSO_MSE[i][j] = franke_LASSO.MSE(z, z_tilde)

	print("|")

	RR = {"OLS" : OLS_RR, "LASSO" : LASSO_RR, "RIDGE" : RIDGE_RR}
	np.savez(path+"RR", **RR)

	MSE = {"OLS" : OLS_MSE, "LASSO" : LASSO_MSE, "RIDGE" : RIDGE_MSE}
	np.savez(path+"MSE", **MSE)
else:
	RR = np.load(path+"RR.npz")
	MSE = np.load(path+"MSE.npz")


path = "figures/"
if noise:
	path += "noise_"
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
plt.plot(n_list, MSE['OLS'])
plt.title(r"OLS $MSE$ on train data")
plt.xlabel(r"n")
plt.ylabel(r"MSE")
#plt.ylim(MSE_min, MSE_max)
plt.savefig(path + "MSE_OLS.png")
if show:
	plt.show()
else:
	plt.close()



# plot RIDGE and LASSO, 3D
cmap_RR = plt.get_cmap("inferno")
cmap_MSE = plt.get_cmap("summer")
Y, X = np.meshgrid(np.log10(lambdas), n_list)
print(X.shape, Y.shape)

for name in ["RIDGE", "LASSO"]:
	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(1, 1, 1)
	plt.title(r"%s $R^2$" %name)
	im = ax.pcolormesh(X, Y, RR[name], cmap=cmap_RR)#, vmin=RR_min, vmax=RR_max)
	plt.xlabel(r"n")
	plt.ylabel(r"$log_{10}(\lambda)$")
	fig.colorbar(im, ax=ax)
	plt.savefig(path + "RR_%s.png" %name)
	if show:
		plt.show()
	else:
		plt.close()

	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(1, 1, 1)
	plt.title(r"%s MSE on train data" %name)
	im = ax.pcolormesh(X, Y, MSE[name], cmap=cmap_MSE)#, vmin=MSE_min, vmax=MSE_max)
	plt.xlabel(r"n")
	plt.ylabel(r"$log_{10}(\lambda)$")
	fig.colorbar(im, ax=ax)
	plt.savefig(path + "MSE_%s.png" %name)
	if show:
		plt.show()
	else:
		plt.close()
