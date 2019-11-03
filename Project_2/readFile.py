import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def readfile():
    cwd = os.getcwd()
    filename = cwd + '/default of credit card clients.xls'
    nanDict = {}
    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
    df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)

    #Remove outliers
    df = df[df.MARRIAGE != 0]
    df = df[df.EDUCATION != 0]
    df = df[df.EDUCATION != 5]
    df = df[df.EDUCATION != 6]

    #Categorize PAY_# columns
    for i in reversed(range(1, 7, 1)):
        df.insert(5, "PAY_" + str(i) + "_CAT", 0)

    columns = [df.PAY_1, df.PAY_2, df.PAY_3, df.PAY_4, df.PAY_5, df.PAY_6]

    for i, p in enumerate(columns):
        df.loc[p <= 0, "PAY_" + str(i+1) + "_CAT"] = p
        df.loc[p <= 0, "PAY_"+ str(i+1)] = 0
        df['PAY_' + str(i+1) + '_CAT'] = df['PAY_' + str(i+1) + '_CAT'].fillna(value=0)

    df1 = df.pop('LIMIT_BAL')
    df2 = df.pop('AGE')
    df3 = df.pop('defaultPaymentNextMonth')
    df['LIMIT_BAL'] = df1
    df['AGE'] = df2
    df['defaultPaymentNextMonth'] = df3

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

    onehotencoder = OneHotEncoder(categories="auto")

    X = ColumnTransformer(
        [("", onehotencoder, np.arange(1, 9, 1)),],
        remainder="passthrough"
        ).fit_transform(X)

    X = X.astype('float32')
    y = y.astype('float32')

    return X, y
