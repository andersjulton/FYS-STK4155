import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from regClass import OLS, LASSO, RIDGE

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def get_data(terrain_data, p):
    ny, nx = np.shape(terrain_data)
    #x, y = np.sort(np.random.uniform(0, 1, nx)), np.sort(np.random.uniform(0, 1, ny))
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    x, y = np.linspace(0, nx/max(nx, ny), nx), np.linspace(0, ny/max(nx, ny), ny)
    print(nx/max(nx, ny), ny/max(nx, ny))
    x, y = np.meshgrid(x, y)
    x, y = np.ravel(x), np.ravel(y)

    #z = terrain_data[0:nx]
    z = terrain_data
    print(np.shape(z))
    z = np.ravel(z)


    print(len(x), len(y), len(z))
    print("X begynner")
    X = OLS(p).CreateDesignMatrix(x, y)
    print("X ferdig")
    return X, z



def plot(terrain):
    plt.imshow(terrain)
    plt.savefig("figures/terrain.pdf")
    plt.show()


def plot_reg(method, filename, p):
    terrain = imread(filename)
    nx, ny = np.shape(terrain)
    X, z = get_data(terrain, p)
    print("ZERO")
    method.p = p
    method.fit(X, z)
    print("ONE")
    new_terrain = method(X)
    print("TWO")
    new_terrain = new_terrain.reshape(nx, ny)
    plt.imshow(new_terrain)
    plt.savefig("figures/terrain_%d.pdf" %p)
    plt.show()





filename = 'data.tif'
terrain = imread(filename)
#plot(terrain)
#print(np.shape(terrain))
#print(np.shape(np.ravel(terrain)))


ols = OLS(p)
ridge = RIDGE(p, 0.001)
lasso = LASSO(p, 0.001)
plot(terrain)
for p in range(3, 15):
    plot_reg(ols, filename, p)





