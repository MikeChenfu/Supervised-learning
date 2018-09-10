#!/usr/bin/env python3

import pandas as pd
import numpy as np
import warnings
import argparse
import matplotlib.pyplot as plt

from func.preprocess import *
from func.balance import *
from func.classifier import *
from func.feature_selection import *

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input file.",required=True)
parser.add_argument("-f", "--method", default='forest', help="feature selection \
                    model: forward, forest, boosting")

args = parser.parse_args()

# read in data
try:
    data = pd.read_csv(args.input, header=None)
    print("Data loaded")
except:
    print("Dataset could not be loaded. Is the dataset missing? Please use -i inputfile")

data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'TARGET']

# preprocess data
data = preprocess(data)

# balance data by SMOTE
X, y, data = handle_imbalanced_data(data)

# drop instance weight from data
instance_weight = data.MARSUPWT
data = data.drop('MARSUPWT',axis=1)
# drop label from data
X = data.drop('TARGET',axis=1)

# feature selection
'''
X : features
y : labels
data : preprocessed data
instance_weight : MARSUPWT column
method: 1:Forward, 2:Random_Forest, 3:Boosting
'''
feature_selection(X,y,data,instance_weight,args.method)














