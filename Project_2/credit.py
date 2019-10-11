import pandas as pd
import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import scikitplot as skplt
from scipy.integrate import simps

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_extraction import DictVectorizer

"""
KLADDEFIL!
"""


np.random.seed(0)
import random
random.seed(0)

def sigmoid(X):
    z = 1/(1 + np.exp(-X))
    return z.astype('float32')

def loss_function(prob, y):
    return (-y*np.log(prob) - (1 - y)*np.log(1 - prob)).mean()

# Reading file into data frame
cwd = os.getcwd()
filename = cwd + '/default of credit card clients.xls'
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

df = df[df.MARRIAGE != 0]

df = df[df.EDUCATION != 0]
df = df[df.EDUCATION != 5]
df = df[df.EDUCATION != 6]


#df = df[df.BILL_AMT1 != 0]
#df = df[df.BILL_AMT2 != 0]
#df = df[df.BILL_AMT3 != 0]
#df = df[df.BILL_AMT4 != 0]
#df = df[df.BILL_AMT5 != 0]
#df = df[df.BILL_AMT6 != 0]
#
#df = df[df.PAY_AMT1 != 0]
#df = df[df.PAY_AMT2 != 0]
#df = df[df.PAY_AMT3 != 0]
#df = df[df.PAY_AMT4 != 0]
#df = df[df.PAY_AMT5 != 0]
#df = df[df.PAY_AMT6 != 0]

X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [2, 3]),],
    remainder="passthrough"
).fit_transform(X)

intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept, X), axis=1)

X = X.astype('float32')
y = y.astype('float32')

y = y.reshape(y.size)
# Train-test split
trainingShare = 0.5
seed  = 1
XTrain, XTest, yTrain, yTest = train_test_split(X, y,
    train_size=trainingShare,
    random_state=seed)

# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

# One-hot's of the target vector
#yTrain, yTest = onehotencoder.fit_transform(yTrain), onehotencoder.fit_transform(yTest)

def fit(X, y):
    lr = 0.01
    theta = np.zeros(X.shape[1])
    for i in tqdm.tqdm(range(10000)):
        z = X @ theta
        prob = sigmoid(z)
        gradient = (X.T @ (prob - y))/y.size
        theta -= lr*gradient
    return theta

#theta = fit(XTrain, yTrain)
#np.save("theta", theta)

theta = np.load("theta.npy")

z = XTrain @ theta
yPred = sigmoid(z)
yPred2 = 1 - yPred
yPred3 = np.array((yPred2, yPred)).T
print(yPred)
ax = skplt.metrics.plot_cumulative_gain(yTrain, yPred3)
lines = ax.lines[1]

defaults = sum(yTrain == 1)
total = len(yTrain)
defaultRate = defaults/total

def bestCurve(defaults, total, defaultRate):
    x = np.linspace(0, 1, total)

    y1 = np.linspace(0, 1, defaults)
    y2 = np.ones(total-defaults)
    y3 = np.concatenate([y1,y2])
    return x, y3

x, best = bestCurve(defaults=defaults, total=total, defaultRate=defaultRate)
plt.plot(x, best)

modelArea = np.trapz(lines.get_ydata(), lines.get_xdata())
bestArea = np.trapz(best, dx = 1/best.size)
ratio = (modelArea - 0.5)/(bestArea - 0.5)
#print(ratio)
#plt.show()
