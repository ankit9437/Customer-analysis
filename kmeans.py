# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:56:58 2020

@author: DELL
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pylab as pl

df=pd.read_csv("Cust_Segmentation.csv")
#print(df.head())

df=df.drop("Address", axis=1)
#print(df.head())

x=df.values[:,1:]
x=np.nan_to_num(x)
clus_data=StandardScaler().fit_transform(x)



k_means=KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means.fit(clus_data)
#print(k_means)
k_means_labels=k_means.labels_
#print(k_means_labels[0:5])
df["Clus_lbl"]=k_means_labels
#print(df["Clus_lbl"])
c0=df[df["Clus_lbl"]==0]
#print(c0)
#print(c0.shape)
c1=df[df["Clus_lbl"]==1]
#print(c1.shape)
c2=df[df["Clus_lbl"]==2]
#print(c2.shape)
#print(c0.describe())
#print(c1.describe())
#print(c2.describe())
#da=pd.DataFrame(df,columns=["Age","Income"])
#df.plot(x="Age",y="Income", kind="scatter")
#plt.show()

pca=PCA(n_components=3).fit(clus_data)
#print(pca)
pca_2d=pca.transform(clus_data)
#print(pca_2d.shape)
pl.figure("Reference plot")
pl.scatter(pca_2d[:,0], pca_2d[:,1], pca_2d[:,2])
kmeans=KMeans(n_clusters=3, random_state=111)

kmeans.fit(clus_data)
pl.figure("K-means with 3 clusters")
pl.scatter(pca_2d[:,0], pca_2d[:,1], pca_2d[:,2],c=kmeans.labels_)

pl.show()