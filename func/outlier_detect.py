import pandas as pd
import numpy as np
import itertools

def outlierDetection(fdata):
    # select the indices for data points you wish to remove
    outliers_lst  = []
    for feature in fdata.columns:
        # calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(fdata.loc[:, feature], 25)
        # calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(fdata.loc[:, feature], 75)
        # use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)
        
        # display the outliers
        #print("Data points considered outliers for the feature '{}':".format(feature))
        
        # finding any points outside of Q1 - step and Q3 + step
        outliers_rows = fdata.loc[~((fdata[feature] >= Q1 - step) & (fdata[feature] <= Q3 + step)), :]
        outliers_lst.append(list(outliers_rows.index))
    
    outliers = list(itertools.chain.from_iterable(outliers_lst))
    # sets are lists with no duplicate entries
    uniq_outliers = list(set(outliers))
    # list of duplicate outliers
    dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))

    #print('Outliers list:\n', uniq_outliers)
    #print('Length of outliers list:\n', len(uniq_outliers))
    
    #print('Duplicate list:\n', dup_outliers)
    #print('Length of duplicates list:\n', len(dup_outliers))
    
    good_data = fdata.drop(fdata.index[dup_outliers]).reset_index(drop = True)
    #print( 'Original shape of data:\n', fdata.shape)
    #print( 'New shape of data:\n', good_data.shape)
    
    return good_data

