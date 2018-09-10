import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def handle_imbalanced_data(data):
    '''
    Handle imbalanced data
       '''
    return up_sampling_SMOTE(data)

# handle imbalanced data by SMOTE
def up_sampling_SMOTE(data):
    y_t = data.TARGET
    X_t = data.drop('TARGET',axis=1)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X_t, y_t)
    
    y_res0 = np.array([y_res])
    total = np.concatenate((X_res,y_res0.T), axis=1)
    
    total = pd.DataFrame(data=total, columns=data.columns)
    return X_res, y_res, total

# check data balance of selected colume 
def check_balance(dataFrame,colName):
    print(dataFrame[colName].value_counts())
