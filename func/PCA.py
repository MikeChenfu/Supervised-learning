import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca(good_data, n, samples):
    # apply PCA by fitting the good data with the same number of dimensions as features
    pca = PCA(n_components=n)
    pca.fit(good_data)
    # transform the sample using the PCA fit above
    pca_samples = pca.transform(samples)
    return pca_results(good_data, pca), pca_samples


# generate PCA results plot
def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    '''
    
    # dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions
    
    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    
    # create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))
    
    # plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    
    # display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))
    
    # save to image
    #fig.savefig("PCA.png")
    
    # return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

