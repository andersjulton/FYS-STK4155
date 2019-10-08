import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from regClass import OLS, LASSO, RIDGE

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from timeit import default_timer as timer
import tqdm

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


fsize = 10
Compute_K_fold = False
Compute_lambdas = False
Compute_Big_X = False
filename = 'data.tif'


def get_data_x_y_z(terrain_data):
    ny, nx = np.shape(terrain_data)
    x, y = np.linspace(0, nx/max(nx, ny), nx), np.linspace(0, ny/max(nx, ny), ny)
    x, y = np.meshgrid(x, y)
    return np.ravel(x), np.ravel(y), np.ravel(terrain_data)


def get_data(terrain_data, p):
    x, y, z = get_data_x_y_z(terrain_data)
    X = OLS(p).CreateDesignMatrix(x, y)
    return X, z

def big_X(terrain_data, p):
    x, y, z = get_data_x_y_z(terrain_data)
    X = OLS(p).CreateDesignMatrix(x, y)
    np.savez("Big_X.npz", *X)

def split_test_train(x, y, z, test_size):
    N = len(x)
    n = int(N*test_size)

    indices = np.linspace(0, N-1, N)
    np.random.shuffle(indices)
    test = np.logical_and(indices >= 0, indices < n)
    train = test == False
    test = x[test], y[test], z[test]
    train = x[train], y[train], z[train]
    return test, train

def plot(terrain):
    plt.imshow(terrain)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    labelleft=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()
    plt.savefig("figures/terrain_reduced.pdf")
    plt.show()


def plot_reg(method, filename, p):
    terrain1 = imread(filename)
    nx1, ny1 = np.shape(terrain1)
    terrain = np.delete(terrain1, 0, -1)
    terrain = np.delete(terrain, -1, 0)
    bsize = 90
    terrain = downscale(terrain, bsize, bsize)
    nx, ny = np.shape(terrain)
    X, z = get_data(terrain, p)
    method.p = p
    method.fit(X, z)
    if Compute_Big_X:
        big_X(terrain1, p)
    Xfull = np.load("Big_X.npz")
    new_terrain = method(Xfull)
    new_terrain = new_terrain.reshape(nx1, ny1)
    plt.imshow(new_terrain)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,
    labelleft=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()
    #plt.savefig("figures/terrain_%d.pdf" %p)
    plt.show()

def find_best_lambda(filename, method):
    terrain = imread(filename)
    terrain = np.delete(terrain, 0, -1)
    terrain = np.delete(terrain, -1, 0)
    bsize = 90
    testTerrain = downscale(terrain, bsize, bsize)
    x, y, z = get_data_x_y_z(testTerrain)
    test, train = split_test_train(x, y, z, 0.2)
    xtest, ytest, ztest = test
    xtrain, ytrain, ztrain = train

    M = 50
    lambdas = np.logspace(-9, -6, M)

    polys = np.arange(4, 18, 1, dtype='int')
    MSE = np.zeros(M)
    points = np.zeros(len(polys))

    for i, p in enumerate(tqdm.tqdm(polys)):
        method.p = p

        Xtrain = method.CreateDesignMatrix(xtrain, ytrain)
        Xtest = method.CreateDesignMatrix(xtest, ytest)

        for j in range(M):
            method.l = lambdas[j]
            method.fit(Xtrain, ztrain)
            MSE[j] = method.MSE(ztest, method(Xtest))

        points[i] = lambdas[np.argmin(MSE)]
    np.savez("terrain_best_lambdas_" + str(method), *points)

def K_fold(method, filename):
    start = timer()

    terrain = imread(filename)
    terrain = np.delete(terrain, 0, -1)
    terrain = np.delete(terrain, -1, 0)
    bsize = 90
    testTerrain = downscale(terrain, bsize, bsize)
    x, y, z = get_data_x_y_z(testTerrain)
    p_list = np.arange(4, 18, 1, dtype=int)
    N = len(p_list)
    lambdas = np.load("terrain_best_lambdas_" + str(method) + ".npz")

    k_fold = np.zeros((N, 2))
    for i in tqdm.tqdm(range(N)):
        method.l = lambdas["arr_" + str(i)]
        method.p = p_list[i]
        k_fold[i] = method.kFoldCV(x, y, z, k=10)
    end = timer()
    print()
    print(str(method), "used %.3g h" %(float(end - start)/3600))
    data = {'R2': k_fold[:, 0], 'MSE' : k_fold[:, 1]}
    np.savez("terrain_kfold_" + str(method), **data)


def downscale(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    newarr = (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    newarr = np.mean(newarr, axis=(1,2))
    return newarr.reshape(int(h/nrows), int(w/ncols))


plot_reg(OLS(13), filename, 13)

if Compute_lambdas:
    find_best_lambda(filename, OLS(0))
    find_best_lambda(filename, LASSO(0, 0))
    find_best_lambda(filename, RIDGE(0, 0))

if Compute_K_fold:
    ols = OLS(0)
    ridge = RIDGE(0, 0.001)
    lasso = LASSO(0, 0.001)
    K_fold(ols, filename)
    K_fold(ridge, filename)
    K_fold(lasso, filename)



"""U, D, V = np.linalg.svd(terrain, full_matrices=False)

start = timer()
beta = V.T @ np.diag(1/D) @ U.T
end = timer()
print("Normal used %.3g h" %(float(end - start)))

Ured = U[:, :25]
Dred = D[:25]
Vred = V[:25, :]

start = timer()
beta = Vred.T @ np.diag(1/Dred) @ Ured.T
end = timer()

print("Reduced used %.3g h" %(float(end - start)))
terrainred = Ured @ np.diag(Dred) @ Vred
#plt.imshow(terrainred, cmap='gray')
#plt.show()"""

if False:
    path = "terrain_kfold_"
    ols = np.load(path+ "OLS.npz")
    ridge = np.load(path + "RIDGE.npz")
    lasso = np.load(path + "LASSO.npz")

    path = "figures/terrain_kfold_"
    p_list = np.arange(4, 18, 1, dtype=int)
    plt.plot(p_list, ols["MSE"], label="OLS")
    plt.plot(p_list, ridge["MSE"], label="RIDGE")
    plt.plot(p_list, lasso["MSE"], label="LASSO")
    plt.legend(fontsize=fsize)
    plt.xticks(p_list, p_list)
    plt.xlabel("polynomial degree", fontsize=fsize)
    plt.ylabel("MSE", fontsize=fsize)
    plt.tight_layout()
    #plt.savefig(path + "MSE.pdf")
    plt.show()

    plt.plot(p_list, ols["R2"], label="OLS")
    plt.plot(p_list, ridge["R2"], label="RIDGE")
    plt.plot(p_list, lasso["R2"], label="LASSO")
    point = np.argmax(ols["R2"])
    plt.scatter(p_list[point], ols["R2"][point], color='r', label="p = {}".format(p_list[point]))
    plt.legend(fontsize=fsize)
    plt.xticks(p_list, p_list)
    plt.xlabel("polynomial degree", fontsize=fsize)
    plt.ylabel(r"$R^2$", fontsize=fsize)
    plt.tight_layout()
    #plt.savefig(path + "R2.pdf")
    plt.show()
