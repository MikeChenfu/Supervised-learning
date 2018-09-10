import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def seg_model(n_clusters, reduce_data, pca_samples, filename, method):
    '''
    Generating segementation model
    Method: 1:Gaussian Mixture Model, 2:K-Means Clustering
    '''
    if method == 'GMM':
        GMModel(n_clusters, reduce_data, pca_samples, filename)
    else:
        KNN(n_clusters, reduce_data, pca_samples, filename)

def GMModel(n_clusters, reduce_data, pca_samples, filename):
    '''
    Gaussian Mixture Model
    Input: 
        n_clusters : number of clusters
        good_data  : dimensionality reduction data
        pca_sample : data for prediction
        filename   : model name you want to save
    '''
    # apply your clustering algorithm of choice to the reduced data
    clusterer = GMM(n_components=n_clusters, n_init=1, init_params='kmeans',
                    verbose=0, random_state=3425 ).fit(reduce_data)
        
    # predict the cluster for each data point
    preds = clusterer.predict(reduce_data)
                    
    # find the cluster centers
    centers = clusterer.means_
                    
    # predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)
    # print(sample_preds)
    # calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduce_data, preds, metric='mahalanobis')
    print("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))

    #save model
    pickle.dump(clusterer, open(filename, 'wb'))

def KNN(n_clusters, reduce_data, pca_samples, filename):
    '''
        K-Means Clustering
        Input:
        n_clusters : number of clusters
        good_data  : dimensionality reduction data
        pca_sample : data for prediction
        filename   : model name you want to save
    '''
    # apply your clustering algorithm of choice to the reduced data
    clusterer = KMeans(n_clusters=n_clusters, init='k-means++',
                       n_init=1, verbose=0, random_state=3425).fit(reduce_data)
        
    # predict the cluster for each data point
    preds = clusterer.predict(reduce_data)
                       
    # find the cluster centers
    centers = clusterer.cluster_centers_
                       
    # predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)
    #print(sample_preds)
                       
    # calculate the mean silhouette coefficient for the number of clusters chosen
    # a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters
    score = silhouette_score(reduce_data, preds, metric='euclidean')
    print( "For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))
    # save model
    pickle.dump(clusterer, open(filename, 'wb'))
