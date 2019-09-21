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


def get_train_data(n):
	x = np.sort(np.random.uniform(0, 1, n))
	y = np.sort(np.random.uniform(0, 1, n))
	x, y = np.meshgrid(x, y)
	z = FrankeFunction(x, y)
	return np.ravel(x), np.ravel(y), np.ravel(z)

def get_test_data(n):
	x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
	z = FrankeFunction(x, y)
	return np.ravel(x), np.ravel(y), np.ravel(z)


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
	ax.scatter(x, y, z, c=z, cmap=cm.coolwarm)



def plot_compare(method):
	ax = plot_frankeFunc(75)
	plot_ML(method, 10, ax)
	plt.show()


if __name__ == "__main__":
	p = 5
	n = 100
	x, y, z = get_train_data(n)

	franke_OLS = OLS(x, y, z, p)
	print("R2 score from OLS: %3.6f" % franke_OLS.RR())
	print("MSE score from OLS: %3.6f\n" % franke_OLS.MSE())
	#plot_compare(franke_OLS)


	l = 0.01
	franke_LASSO = LASSO(x, y, z, p, l)
	print("R2 score from LASSO: %3.6f" % franke_LASSO.RR())
	print("MSE score from LASSO: %3.6f\n" % franke_LASSO.MSE())
	#plot_compare(franke_LASSO)

	l = 0.01
	franke_RIDGE = RIDGE(x, y, z, p, l)
	print("R2 score from RIDGE: %3.6f" % franke_RIDGE.RR())
	print("MSE score from RIDGE: %3.6f\n" % franke_RIDGE.MSE())
	#plot_compare(franke_RIDGE)
