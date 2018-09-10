import pandas as pd
import numpy as np
import pickle
from random import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def classifier_method(X,y,data,instance_weight,method,filename, modelname):
    '''
    Kfold cross validation and Bootstrap
    '''
    if method == 'kfold':
        kfold_cross_validation(X,y, data)
    else:
        bootstrap(data, instance_weight,filename, modelname)

def kfold_cross_validation(x,y, data):
    '''
    Kfold cross validation
    Include three models 'LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING'
    '''
    model_names = ['LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING']
    for model_name in model_names:
        if(model_name == 'LOGISTIC REGRESSION'):
            model = LogisticRegression()
        elif(model_name == 'RANDOM FORESTS'):
            model = RandomForestClassifier(n_estimators = 100, max_features='auto', min_samples_split=2)
        else:
            model = AdaBoostClassifier(n_estimators = 50, learning_rate =1.0)
        # apply model of choice to cross validation
        #a = cross_val_score(model, x, y, cv=10, scoring='roc_auc')
        a = cross_val_score(model, x, y, cv=10, scoring='accuracy')
        mean_score = a.mean()
        print("AUC by cross validation:",model_name, mean_score)
        print('CV %s AUC_mean: %.4f' % (model_name, mean_score))


def bootstrap(data, instance_weight,filename, modelname):
    '''
    Bootstrp method
    Include three models 'LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING'
    '''
    # configure bootstrap
    n_iterations = 100
    values = data.values

    if(modelname == 'logistic'):
        model = LogisticRegression()
    elif(modelname == 'forest'):
        model = RandomForestClassifier(n_estimators = 100, max_features='auto', min_samples_split=2)
    else:
        model = AdaBoostClassifier(n_estimators = 50, learning_rate =1.0)
    # run bootstrap
    stats = list()
    auc_stats = list()
    max_score =0;
    for i in range(n_iterations):
        # prepare train and test sets
        random_num = randint(1, 10000)
        weight_bs = resample(instance_weight,n_samples=100, random_state=random_num )
        train_bs = resample(values, n_samples=100,random_state=random_num)
        test_bs = np.array([x for x in values if x.tolist() not in train_bs.tolist()])
            
        y = train_bs[:,-1]
        x = train_bs[:,:-1]
        y_te = test_bs[:,-1]
        x_te = test_bs[:,:-1]
        # fit model with instance weight
        model = model.fit(x,y, sample_weight = weight_bs)
        # evaluate model
        pred = model.predict(x_te)
        score = accuracy_score(y_te, pred)
        # find optiml model
        if score > max_score:
            max_score = score
            good_model = model
        stats.append(score)
        auc =  roc_auc_score(y_te, pred)
        auc_stats.append(auc)
        print('Accuracy:',score,' AUC:',auc)
        
    logreg_auc_mean = np.mean(auc_stats)
    logreg_accuracy_mean = np.mean(stats)
        
    print('Bootstrap %s AUC_mean: %.4f' % (modelname, logreg_auc_mean))
    print('Bootstrap %s Accuracy_mean: %.4f' % (modelname, logreg_accuracy_mean))
    # save model
    pickle.dump(good_model, open(filename, 'wb'))




