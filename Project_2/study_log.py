import numpy as np
import tqdm
from readFile import *
import matplotlib.pyplot as plt
import scikitplot as skplt
from logClass import GradientDescent

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

X, y = readfile()

trainingShare = 0.5
seed  = 1
XTrain, XTest, yTrain, yTest = train_test_split(X, y,
    train_size=trainingShare,
    random_state=seed)

# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

gradDesc = GradientDescent(max_iter=10000)

gradDesc.fit(XTrain, yTrain)

yPredTest = gradDesc(XTest)
yPredTrain = gradDesc(XTrain)

print(gradDesc.get_Area_ratio(yTrain, yPredTrain))
print(gradDesc.get_Area_ratio(yTest, yPredTest))
