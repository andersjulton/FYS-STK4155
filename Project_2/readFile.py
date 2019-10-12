import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Reading file into data frame
def readfile():
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

    return X, y
