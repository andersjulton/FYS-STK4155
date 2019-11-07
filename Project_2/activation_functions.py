import numpy as np

'''
Collection of activation functions
'''

class ActivationFunction(object):
	def __init__(self, f, df, a=0.01, name):
		self.a = a
		self._f = f
		self._df = df
		self.name = name 

	def __call__(self, z):
		return self._f(z)

	def derivative(self, z):
		return self._df(z)

	def __str__(self):
		return self.name


# sigmoid function: z in (0, 1)
def f_sigmoid(z):
	return 1/(1 + np.exp(-z))

def df_sigmoid(z):
	sigm = 1/(1 + np.exp(-z))
	return sigm*(1 - sigm)

sigmoid = ActivationFunction(f_sigmoid, df_sigmoid, "sigmoid")

# ReLu function: z in [0, inf)
def f_ReLU(z):
	zn = np.zeros(z.shape)
	indices = np.where(z > 0)
	zn[indices] = z[indices]
	return zn

def df_ReLU(z):
	zn = np.zeros(z.shape)
	indices = np.where(z >= 0)
	zn[indices] = 1
	return zn

ReLU = ActivationFunction(f_ReLU, df_ReLU, "ReLU")


# PReLU function: z in (-inf, inf)
def f_PReLU(z):
	zn = z*self.a
	indices = np.where(z > 0)
	zn[indices] = z[indices]
	return zn

def df_PReLU(z):
	zn = np.zeros(z.shape) + self.a
	zn[z >= 0] = 1
	return zn

PReLU = ActivationFunction(f_PReLU, df_PReLU, "PReLU")


# tanh function: z in (-1, 1)
df_tanh = lambda z:  1 - np.tanh(z)**2

tanh = ActivationFunction(np.tanh, df_tanh, "tanh")


# identity function: z in (-inf, inf)
f_identity = lambda z: z
df_identity = lambda z: 1

identity = ActivationFunctions(f_identity, df_identity, "identity")


