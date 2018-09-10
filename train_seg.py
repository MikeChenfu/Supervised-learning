#!/usr/bin/env python3
import pandas as pd
import numpy as np
import warnings
import argparse
import matplotlib.pyplot as plt

from func.preprocess import *
from func.balance import *
from func.PCA import *
from func.feature_relevance import *
from func.outlier_detect import *
from func.seg_model import *

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input file.", required=True)
parser.add_argument("-cluster", "--cluster", default='KNN', help="segementation model: KNN or GMM")
parser.add_argument("-o", "--output", default='seg_model.sav', help="output model")
parser.add_argument("-n", "--ncluster",type=int, default=10, help="number of clusters ")

args = parser.parse_args()

# read in data
try:
    data = pd.read_csv(args.input, header=None)
    print("Data loaded")
except:
    print("Dataset could not be loaded. Is the dataset missing?")

data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'TARGET']

# features for segementation model
filterCol = ['AAGE', 'AHGA', 'ASEX', 'CAPGAIN',
             'CAPLOSS','DIVVAL', 'NOEMP','WKSWORK']

# preprocess data
data = preprocess(data)

# balance data
X, y, data = handle_imbalanced_data(data)

# filtered data
fdata = data[filterCol]
#smaller amount of data for training because of memory limitation
fdata =fdata.loc[0:30000]

#define several samples
indices = [100, 200, 300, 400, 500]
samples = pd.DataFrame(fdata.loc[indices], columns = fdata.columns).reset_index(drop = True)

# Feature Relevance
#R2_score(fdata)
#plot_corr(fdata)

#Outlier Detection
good_data = outlierDetection(fdata)

#principal component analysis
#pca_results, pca_samples = pca(good_data, 8, samples)
#print(pca_results['Explained Variance'].cumsum())

#92.80% of the variance explained by the first and second principal components.
#Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2)
pca.fit(good_data)
#Transform the good data and samples using the PCA
reduced_data = pca.transform(good_data)
pca_samples = pca.transform(samples)
# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
pca_samples = pd.DataFrame(pca_samples, columns = ['Dimension 1', 'Dimension 2'])

# generating segementation model
'''
nclusters : number of clusters
reduce_data  : dimensionality reduction data
pca_sample : data for prediction
output   : model name you want to save
cluster model:  Gaussian Mixture Model and K-Means Clustering
'''

seg_model(args.ncluster, reduced_data, pca_samples, args.output, args.cluster)


