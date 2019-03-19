import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn import random_projection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler

from assignment4_NN import NN
from assignment4_NN import error_rate

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 

#####Creating Dummies###########
def createDummy(data):
    data = pd.get_dummies(data, drop_first=True)
    return data

def normalize(X):
	scaler = MinMaxScaler()
	X=scaler.fit_transform(X)
	return X

def elbow(X):
	Nc = range(1, 20)
	kmeans = [KMeans(n_clusters=i) for i in Nc]
	kmeans
	score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
	score
	pl.plot(Nc,score)
	pl.xlabel('Number of Clusters')
	pl.ylabel('Score')
	pl.title('Elbow Curve')
	pl.show()

def cluster_kmeans(X):
    kmeans = KMeans(n_clusters=5) 
    kmeans.fit(X)
    kmeans_clusters = kmeans.predict(X)
    return kmeans_clusters

def em_plot(X):
    n_components = np.arange(2, 30)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('clusters')
    plt.ylabel('AIC/BIC')
    plt.show()

def em(X):
	gmm = GaussianMixture(5, covariance_type='full', random_state=0).fit(X)
	gmm_clusters = gmm.predict(X)
	return gmm_clusters

def pca_transformed(X):
    pca = PCA(n_components=2, svd_solver='full')
    pca = pca.fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    return pca_transformed

def ica(X):
	ica = FastICA(random_state=0)
	ica_transformed = ica.fit_transform(X) 

def rca(X):
	rca = random_projection.SparseRandomProjection()
	rca_projected = rca.fit_transform(X)

def feature_selection(X, y):
	clf = ExtraTreesClassifier(n_estimators=50)
	clf = clf.fit(X, y)
	print(clf.feature_importances_)
	model = SelectFromModel(clf, prefit=True)
	tree_selected_features = model.transform(X)
	print(tree_selected_features.shape)         

### import and preprocessing
dataset = 'C:/Users/kohul/OneDrive/Documents/ML/student-mat.csv'
data = pd.read_csv(dataset, sep=';')
y = data.iloc[:,32:33].values
y = (np.where(y>10,1,0))
data = data.drop(['G1','G2','G3'],axis=1)
X = createDummy(data)
X = normalize(X)
##### Cluster #####
### K means
elbow(X)
kmeans_clusters = cluster_kmeans(X)
### GMM
em_plot(X)
gmm_clusters = em(X)

##### Dimentionality Reduction ######
### PCA
pca_X = pca_transformed(X)
### ICA
#ica(X)
### RCA
#rca(X)
### Feature Selection
feature_selection(X,y)

##### Cluster with Dimentionality Reduced dataset #####
### K Means
#elbow(pca_X)
#kmeans_clusters = cluster_kmeans(pca_X)

### NN with Dim Reduction
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
#NN(Xtrain, Xtest, Ytrain, Ytest)

### NN with clustering

print(X)
#if __name__ == '__main__':
#	main()


