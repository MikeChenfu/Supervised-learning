import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

def feature_selection(X,y,data,weight,method):
    '''
    Handle imbalanced data
    Method: 1:Forward_Feature_Selection, 2:Random_Forest, other:Boosting_Feature_Selection
    '''
    if method == 'forward':
        forward_FSS(X, y, data)
    elif method == 'forest':
        random_forest_FSS(X, y, data, weight)
    else:
        boosting_feature_selection(X, y, data, weight)

def forward_FSS(x,y, fdata):
    '''
    Forward Feature Selection
    '''
    features = fdata.columns[0:-1]
    logreg = LogisticRegression()
    max_auc = -1
    max_auc_label = ''
    selected = []
    # step 1: Select the first feature
    for f in features:
        x = fdata[f]
        x = x.values.reshape(-1, 1)
        average_auc = cross_val_score(logreg, x, y, cv=3, scoring='roc_auc').mean()
        if(average_auc > max_auc):
            max_auc = average_auc
            max_auc_label = f;
    print(max_auc_label)
    print(max_auc)
    selected.append(max_auc_label)
    features = features.drop(labels=max_auc_label)

    # in each iteration, select one feature
    while(len(features) > 0):
        max_auc = -1
        for f in features:
            x = fdata[np.concatenate((selected, [f]))]
            average_auc = cross_val_score(logreg, x, y, cv=3, scoring='roc_auc').mean()
            if(average_auc > max_auc):
                max_auc = average_auc
                max_auc_label = f;
        print(max_auc_label)
        print(max_auc)
        selected.append(max_auc_label)
        features = features.drop(labels=max_auc_label)

def random_forest_FSS(x, y, rdata, instance_weight):
    '''
    Random Forest for Feature Selection
    '''
    rf = RandomForestClassifier(n_estimators = 10, max_features="auto", min_samples_split=2)
    # add instance weight in to classfier model
    rf.fit(x, y, sample_weight = instance_weight)
    rf.feature_importances_.reshape(-1, 1)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    
    # print the feature ranking
    print("Feature ranking:")
    feature_cols = rdata.columns[0:-1]
    for f in range(x.shape[1]):
        print(feature_cols[indices[f]], importances[indices[f]])


def boosting_feature_selection(x, y, bdata,instance_weight ):
    '''
    Boosting Feature Selection
    '''
    boost = AdaBoostClassifier(n_estimators = 50, learning_rate = 1)
    # add instance weight in to classfier model
    boost.fit(x, y, sample_weight = instance_weight)
    test_auc = cross_val_score(boost, x, y, cv=5, scoring='roc_auc').mean()
    
    importances = boost.feature_importances_
    std = np.std([tree.feature_importances_ for tree in boost.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    # print the feature ranking
    print("Feature ranking:")
    feature_cols = bdata.columns[0:-1]
    for f in range(x.shape[1]):
        print(feature_cols[indices[f]], importances[indices[f]])

