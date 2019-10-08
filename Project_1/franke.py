from regClass import OLS, LASSO, RIDGE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def FrankeFunction(x, y):
	term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
	term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
	term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
	term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
	return term1 + term2 + term3 + term4


def get_noise(n):
	return np.random.normal(loc=0.0, scale=1, size=n)


def get_train_data(n, noise=False):
	x = np.sort(np.random.uniform(0, 1, n))
	y = np.sort(np.random.uniform(0, 1, n))
	x, y = np.meshgrid(x, y)
	f = np.ravel(FrankeFunction(x, y))
	if noise:
		z = f + get_noise(n*n)
	else:
		z = f
	return np.ravel(x), np.ravel(y), z


def get_test_data(n, noise=False):
	x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
	f = np.ravel(FrankeFunction(x, y))
	if noise:
		z = f + get_noise(n*n)
	else:
		z = f
	return np.ravel(x), np.ravel(y), z

def get_test_train_data(N, test_size, noise=False):
	x, y = np.sort(np.random.uniform(0, 1, N)), np.sort(np.random.uniform(0, 1, N))
	x, y = np.meshgrid(x, y)
	f = np.ravel(FrankeFunction(x, y))
	n = int(len(f)*test_size)

	indices = np.linspace(0, len(f)-1, len(f))
	np.random.shuffle(indices)
	test = np.logical_and(indices >= 0, indices < n)
	train = test == False

	if noise:
		z = f + get_noise(len(f))
		ztest = z[test]
		ztrain = z[train]
	else:
		ztrain = f[train] + get_noise(len(f))[train]
		ztest = f[test]

	test = (np.ravel(x)[test], np.ravel(y)[test], ztest)
	train = (np.ravel(x)[train], np.ravel(y)[train], ztrain)

	return test, train

def plot_frankeFunc(n):
	fig = plt.figure()
	x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
	z = FrankeFunction(x, y)
	ax = fig.gca(projection='3d')
	# Plot the surface.
	surf = ax.plot_surface(x, y, z, cmap="coolwarm", linewidth=0, antialiased=False, alpha=0.5)

	# Customize the z axis.
	ax.set_zlim(-0.10, 1.40)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	return ax


def plot_ML(method, n, ax):
	x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
	x, y = np.ravel(x), np.ravel(y)
	z = method(x, y)
	ax.scatter(x, y, z, c=z, cmap="coolwarm")



def plot_compare(method, n):
	ax = plot_frankeFunc(n)
	plot_ML(method, n, ax)
	plt.show()


if __name__ == "__main__":
	p = 5
	n = 100
	x, y, z = get_train_data(n)

	franke_OLS = OLS(p)
	franke_OLS.learn(z, x, y)
	plot_compare(franke_OLS)


	l = 0.01
	franke_LASSO = LASSO(p, l)
	franke_LASSO.learn(z, x, y)
	plot_compare(franke_LASSO)

	l = 0.01
	franke_RIDGE = RIDGE(p, l)
	franke_RIDGE.learn(z, x, y)
	plot_compare(franke_RIDGE)
