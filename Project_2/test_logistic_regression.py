import numpy as np
from numpy import random as rand
from logClass import *


def test_step_function(method, limit):
	''' 
		Testing trivial problem: is number higher or lower than the limit?
	'''
	N = 1000
	edge = 100
	# Train data
	X_train = np.ones((N, 2))
	y_train = np.ones(N, dtype=int)
	X_train[:, 1] = 2*edge*rand.rand(N) - edge
	y_train[X_train[:, 1] < limit] = 0

	# Learn
	log_reg = method()
	log_reg.fit(X_train, y_train)

	# Test data
	N = 100
	X = np.ones((N, 2))
	y = np.ones(N, dtype=int)
	X[:, 1] = np.linspace(-edge, edge, N)
	y[X[:, 1] < limit] = 0

	# Predict
	y_pred = log_reg(X)


	accuracy = log_reg.accuracy(y, y_pred)
	print(accuracy)
	assert accuracy > 0.9, "Testing step function failed for " + str(method)


def test_all(method):
	print(f"Testing {method}")
	test_step_function(method, 0)
	test_step_function(method, 50)
	test_step_function(method, -30)




test_all(GradientDescent)
test_all(StochasticGradient)
test_all(StochasticGradientMiniBatch)
test_all(ADAM)


