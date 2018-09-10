import pandas as pd
import numpy as np
from sklearn import preprocessing

def preprocess(data):
    '''
    Preprocess the input data
    Includes clean null data and combine data
    Encode and normalize data
    '''
    # replace ? with NaN
    data = data.replace('?', np.NaN)
    # set label value
    if 'TARGET' in data.columns:
        data['TARGET'] = [ 1 if income == ' 50000+.' else 0 for income in data.TARGET ]

    # bin AAGE to 0-30 31-60 61 -90
    bins = [0, 30, 60, 90]
    groupNames = ["young","adult","old"]
    data['AAGE'] = pd.cut(data['AAGE'], bins, labels = groupNames)

    # bin numeric variables with Zero and MoreThanZero
    num_bin_list = ['AHRSPAY', 'CAPGAIN', 'CAPLOSS','DIVVAL']
    for column in num_bin_list:
        data[column]=data[column].apply(lambda x: checkGtZero(x))

    # encode categorical data
    for column in data.columns:
        encodeCategoricalData(data,column)

    # normalize 'NOEMP' and 'WKSWORK'
    data = normalizeData(data)
    return data

def checkGtZero(val):
    if(val==0):
        return 'Zero'
    else:
        return 'MoreThanZero'

def encodeCategoricalData(df,column):
    le = preprocessing.LabelEncoder()
    df[column]=le.fit_transform(df[column].astype(str))

def normalization(nData, column):
    nData[column] = preprocessing.scale(np.sqrt(nData[column]))
    return nData

def normalizeData(nData):
    nData = normalization(nData, 'NOEMP')
    nData = normalization(nData, 'WKSWORK')
    return nData
