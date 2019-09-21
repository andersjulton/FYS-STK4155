from regClass import OLS, LASSO, RIDGE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


compute_RR = False

def frankeFunc(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4



p = 5
franke = Regression(p, l=0.01, f=frankeFunc, n=100)

'''
franke.LASSO()
print("R2 score from LASSO: %3.6f" % franke.R2())
print("MSE score from LASSO: %3.6f\n" % franke.MSE())

franke.OLS()
print("R2 score from OLS: %3.6f" % franke.R2())
print("MSE score from OLS: %3.6f\n" % franke.MSE())

franke.RIDGE()
print("R2 score from RIDGE: %3.6f" % franke.R2())
print("MSE score from RIDGE: %3.6f\n" % franke.MSE())
'''

n = [10, 50, 75, 100]
if compute_RR:
	LASSO = np.zeros(len(n))
	OLS = np.zeros(len(n))
	RIDGE = np.zeros(len(n))

	for i, j in enumerate(n):
		print(f"computing for n = {j}.")
		franke = Regression(p, l=0.01, f=frankeFunc, n=j)
		franke.LASSO()
		LASSO[i] = franke.R2()
		franke.OLS()
		OLS[i] = franke.R2()
		franke.RIDGE()
		RIDGE[i] = franke.R2()

	RR = {"LASSO" : LASSO, "OLS" : OLS, "RIDGE" : RIDGE}
	np.savez("RR", **RR)
else:
	RR = np.load("RR.npz")
	LASSO = RR["LASSO"]
	OLS = RR["OLS"]
	RIDGE = RR["RIDGE"]

font = 15
plt.plot(n, LASSO, label="LASSO")
plt.plot(n, OLS, label="OLS")
plt.plot(n, RIDGE, label="RIDGE")
plt.legend(fontsize=font)
plt.xlabel(r"$n$", fontsize=font)
plt.ylabel(r"$R^2$", fontsize=font)
plt.show()
