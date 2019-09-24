import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sklearn.linear_model as skl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


# Load the terrain
terrain1 = imread('data.tif')
# Show the terrain


def CreateDesignMatrix_X(x, y, n = 5):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1)*(n + 2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i)*(i + 1)/2)
        for k in range(i + 1):
            X[:, q + k] = x**(i - k)*y**k
    return X

def OLS(X, z):
    #U, s, VT = np.linalg.svd(X)
    #D = np.diag(s**2)
    #Xinv = np.linalg.inv(VT.T @ D @ VT)
    lin = skl.LinearRegression().fit(X, z)
    beta = lin.coef_
    #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
    #beta = Xinv @ X.T @ z
    ztilde = X @ beta
    return ztilde, beta

def RIDGE(X, z, l):

    clf_r = skl.Ridge(alpha = l).fit(X, z)
    beta = clf_r.coef_
    ztilde = X @ beta
    return ztilde, beta

def r2Score(y, ytilde):
    return 1 - np.sum(y - ytilde)**2/np.sum(y - np.mean(y))**2

def MSE(y, ytilde):
    return np.mean((y - ytilde)**2)

ny, nx = np.shape(terrain1)

x, y = np.sort(np.random.uniform(0, 1, nx)), np.sort(np.random.uniform(0, 1, nx))
x, y = np.meshgrid(x, y)
xr, yr = np.ravel(x), np.ravel(y)
X = CreateDesignMatrix_X(xr, yr)


terr = terrain1[0:nx]


z = np.ravel(terr)

#ztilde, beta = OLS(X, z)
ztilde, beta = RIDGE(X, z, 0.1)

zt = ztilde.reshape(nx, nx)

fig = plt.figure()
ax1 = fig.add_subplot(211)
surf = ax1.imshow(zt, cmap="gray", extent = [0, 55.6, 0, 55.6])
#ax1.imshow(zt, cmap='gray', extent = [0, 55.6, 0, 55.6])
ax1.set_xlabel('Km')
scalebar = AnchoredSizeBar(ax1.transData,
                           10, '10 km', 'lower right')

ax1.add_artist(scalebar)

ax2 = fig.add_subplot(212)
ax2.imshow(terrain1[0:nx], cmap='gray', extent = [0, 55.6, 0, 55.6])

plt.show()
