import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor

def R2_score(fdata):
    # create loop to test each feature as a dependent variable
    for col in fdata.columns:
        new_data = fdata.drop([col], axis = 1)
        new_feature = pd.DataFrame(fdata.loc[:, col])
        
        # split the data into training and testing sets using the given feature as the target
        X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.25, random_state=42)
        # create a decision tree regressor and fit it to the training set
        dtr = DecisionTreeRegressor(random_state=42)
        # fit
        dtr.fit(X_train, y_train)
        score = dtr.score(X_test, y_test)
        print('R2 score for {} as dependent variable: {}'.format(col, score))

def plot_corr(df,size=10):
    '''
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.
        
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot
    '''
    fig = plt.figure(figsize = (20, 15))
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(df, interpolation='nearest')
    ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    fig.savefig("correlation.png")
