import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Reading file into data frame
def readfile(change_values=False, remove_values=False):
    cwd = os.getcwd()
    filename = cwd + '/default of credit card clients.xls'
    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    df = df[df.MARRIAGE != 0]
    df = df[df.EDUCATION != 0]
    df = df[df.EDUCATION != 5]
    df = df[df.EDUCATION != 6]

    if change_values:
        """
        Manual inspection showed that many of the -2 values in PAY_# should be 0.
        """

        df[df.PAY_0 == -2] = 0
        df[df.PAY_2 == -2] = 0
        df[df.PAY_3 == -2] = 0
        df[df.PAY_4 == -2] = 0
        df[df.PAY_5 == -2] = 0
        df[df.PAY_6 == -2] = 0

    if remove_values:
        df = df[df.PAY_0 != -2]
        df = df[df.PAY_2 != -2]
        df = df[df.PAY_3 != -2]
        df = df[df.PAY_4 != -2]
        df = df[df.PAY_5 != -2]
        df = df[df.PAY_6 != -2]

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

    onehotencoder = OneHotEncoder(categories="auto")

    X = ColumnTransformer(
        [("", onehotencoder, [2, 3]),],
        remainder="passthrough"
        ).fit_transform(X)

    intercept = np.ones((X.shape[0], 1))
    #X = np.concatenate((intercept, X), axis=1)

    X = X.astype('float32')
    y = y.astype('float32')

    return X, y
