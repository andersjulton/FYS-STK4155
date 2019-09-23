import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from regClass import

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def get_train_data(terrain_data):
    ny, nx = np.shape(terrain_data)
    x, y = np.sort(np.random.uniform(0, 1, nx)), np.sort(np.random.uniform(0, 1, ny))
    x, y = np.meshgrid(x, y)
    z = terrain1[0:nx]
    return np.ravel(x), np.ravel(y), np.ravel(z)



terrain = imread('data.tif')
x, y, z = get_data(terrain)







zt = ztilde.reshape(nx, nx)

def plot():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.imshow(zt, cmap='gray', extent = [0, 55.6, 0, 55.6])
    ax1.set_xlabel('Km')
    scalebar = AnchoredSizeBar(ax1.transData,
                               10, '10 km', 'lower right')

    ax1.add_artist(scalebar)
    ax2 = fig.add_subplot(212)
    ax2.imshow(terrain1[0:nx], cmap='gray', extent=[0, 55.6, 0, 55.6])
    plt.show()