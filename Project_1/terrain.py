import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from regClass import OLS, LASSO, RIDGE

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from timeit import default_timer as timer

fsize = 10
Compute_K_fold = True

def get_data_x_y_z(terrain_data):
    ny, nx = np.shape(terrain_data)
    x, y = np.linspace(0, nx/max(nx, ny), nx), np.linspace(0, ny/max(nx, ny), ny)
    x, y = np.meshgrid(x, y)
    return np.ravel(x), np.ravel(y), np.ravel(terrain_data)
 

def get_data(terrain_data, p):
    x, y, z = get_data_x_y_z(terrain_data)
    X = OLS(p).CreateDesignMatrix(x, y)
    return X, z





def plot(terrain):
    plt.imshow(terrain)
    plt.savefig("figures/terrain.pdf")
    plt.show()


def plot_reg(method, filename, p):
    terrain = imread(filename)
    nx, ny = np.shape(terrain)
    X, z = get_data(terrain, p)
    method.p = p
    method.fit(X, z)
    new_terrain = method(X)
    new_terrain = new_terrain.reshape(nx, ny)
    plt.imshow(new_terrain)
    plt.savefig("figures/terrain_%d.pdf" %p)
    plt.show()


def K_fold(method, filename):
    start = timer()
    terrain = imread(filename)
    x, y, z = get_data_x_y_z(terrain)
    p_list = np.arange(2, 12, 1, dtype=int)
    N = len(p_list)
    k_fold = np.zeros((N, 2))
    for i in range(N):
        print(p_list[i], end=" ", flush=True)
        method.p = p_list[i]
        k_fold[i] = method.kFoldCV(x, y, z, k=10)
    end = timer()
    print()
    print(str(method), "used %.3g h" %(float(end - start)/3600))
    data = {'R2': k_fold[:, 0], 'MSE' : k_fold[:, 1]}
    np.savez("store_results/terrain_kfold_" + str(method), **data)






filename = 'data.tif'
terrain = imread(filename)
#plot(terrain)
#print(np.shape(terrain))
#print(np.shape(np.ravel(terrain)))


if Compute_K_fold:
    ols = OLS(0)
    ridge = RIDGE(0, 0.001)
    lasso = LASSO(0, 0.001)
    K_fold(ols, filename)
    K_fold(ridge, filename)
    K_fold(lasso, filename)


path = "store_results/terrain_kfold_"
ols = np.load(path+ "OLS.npz")
ridge = np.load(path + "RIDGE.npz")
lasso = np.load(path + "LASSO.npz")

path = "figures/terrain_kfold_" 
p_list = np.arange(2, 12, 1, dtype=int)
plt.plot(p_list, ols["MSE"], label="OLS")
plt.plot(p_list, ridge["MSE"], label="RIDGE")
plt.plot(p_list, lasso["MSE"], label="LASSO")
plt.legend(fontsize=fsize)
plt.xticks(p_list, p_list)
plt.xlabel("polynomial degree", fontsize=fsize)
plt.ylabel("MSE", fontsize=fsize)
plt.tight_layout()
plt.savefig(path + "MSE.pdf")
plt.show()

plt.plot(p_list, ols["R2"], label="OLS")
plt.plot(p_list, ridge["R2"], label="RIDGE")
plt.plot(p_list, lasso["R2"], label="LASSO")
plt.legend(fontsize=fsize)
plt.xticks(p_list, p_list)
plt.xlabel("polynomial degree", fontsize=fsize)
plt.ylabel(r"$R^2$", fontsize=fsize)
plt.tight_layout()
plt.savefig(path + "R2.pdf")
plt.show()








