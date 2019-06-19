# -*- coding: utf-8 -*-
import time

import pandas as pd
import numpy as np

from sklearn import cluster
from sklearn import metrics

from math import floor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

#
#
#       FAMILIAS NUMEROSAS
#       
#
#
#       VARIABLES USADAS: NUMERO DE HIJOS, REGIMEN DE TENENCIA DEL HOGAR,
#                         SUPERFICIE UTIL DEL HOGAR, NUMERO DE HABITACIONES
#       
#

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


def calculateMeanDictionary(cluster,cluster_col = 'cluster'):
    vars = list(cluster)
    vars.remove(cluster_col)
    return dict(np.mean(cluster[vars],axis=0))


def calculateDeviationDictionary(cluster, cluster_col = 'cluster'):
    vars = list(cluster)
    vars.remove(cluster_col)
    return dict(np.std(cluster[vars],axis=0))


def makeScatterPlot(data,outputName=None,displayOutput=True):
    sns.set()
    variables = list(data)
    variables.remove('cluster')
    sns_plot = sns.pairplot(data, vars=variables, palette='Paired', plot_kws={"s": 25}, diag_kind="hist")  # en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)

    sns.pairplot(data, vars=variables, hue="cluster") #en hue indicamos que la columna 'cluster' define los colores

    if displayOutput:
        plt.show()
        
        
def createMeanClusterDF(dataFrame, clusterCol = 'cluster'):
    n_clusters = list(set(dataFrame[clusterCol]))

    my_mean_df = pd.DataFrame()
    my_deviation_df = pd.DataFrame()

    for cluster_n in n_clusters:
        my_cluster = dataFrame[dataFrame[clusterCol] == cluster_n]
        meanDic = calculateMeanDictionary(cluster=my_cluster,cluster_col = clusterCol)
        deviationDic = calculateDeviationDictionary(cluster=my_cluster, cluster_col = clusterCol)
        stdDF = pd.DataFrame(deviationDic, index=[str(cluster_n)])
        auxDF = pd.DataFrame(meanDic,index=[str(cluster_n)])
        my_mean_df = pd.concat([my_mean_df,auxDF])
        my_deviation_df = pd.concat([my_deviation_df,stdDF])

    return [my_mean_df, my_deviation_df]


def findIndex(iterable, value):
    index = -1
    for val in iterable:
        if(value == val):
            return index

    return index


def makeHeatmap(data,displayOutput=True,outputName=None):
    
    meanDF, stdDF = createMeanClusterDF(dataFrame=data)
    anotations = True
    sns.heatmap(data=meanDF, linewidths=.2, cmap='YlGnBu', annot=anotations, fmt='.4f', xticklabels='auto')
    plt.xticks(rotation=0)

    if displayOutput:
        plt.show()
        
censo = pd.read_csv('censo_granada.csv')
censo = censo.replace(np.NaN,0) #los valores en blanco realmente son otra categoría que nombramos como 0

#       FAMILIAS NUMEROSAS
subset = censo.loc[censo['FAMNUM']==2]

#seleccionar variables de interés para clustering
usadas = ['NHIJO', 'TENEN', 'SUT', 'NHAB']
X = subset[usadas]
print(X.shape)
#selecciona 1000 instancias del caso de uso
X = X.sample(2500, random_state=123456)
#normaliza
X_normal = X.apply(norm_to_zero_one)

print("Tamaño del conjunto de datos original")
print(censo.shape)
print("Tamaño del conjunto de datos en este caso de uso")
print(X.shape)

k_means = cluster.KMeans(init='k-means++', n_clusters=7, n_init=5)
mbkm = cluster.MiniBatchKMeans(n_clusters = 5)
ms = cluster.MeanShift()
affinity_propagation = cluster.AffinityPropagation()
dbscan = cluster.DBSCAN(eps = 0.3)
ward = cluster.AgglomerativeClustering(n_clusters = 4, linkage = 'ward')

clustering_algorithms = (
        ('K-means', k_means),
        ('MiniBatchKMeans', mbkm),
        ('MeanShift', ms),
        ('AffinityPropagation', affinity_propagation),
        ('DBSCAN', dbscan),
        ('Ward', ward)
)

cluster_predict = {}
k = {}
metric_CH = {}
metric_SC = {}

for name, algorithm in clustering_algorithms:
    print('{:19s}'.format(name), end='')
    t = time.time()
    cluster_predict[name] = algorithm.fit_predict(X_normal)
    tiempo = time.time()- t
    k[name] = len(set(cluster_predict[name]))
    print(": k: {:3.0f},".format(k[name]),end='')
    print("{:6.2f} segundos, ".format(tiempo),end='')
    if (k[name]>1):
        metric_CH[name] = metrics.calinski_harabaz_score(X_normal, cluster_predict[name])
        metric_SC[name] = metrics.silhouette_score(X_normal, cluster_predict[name], metric='euclidean', sample_size=floor(0.1*len(X)), random_state=123456)

    print("CH index: {:9.3f}, ".format(metric_CH[name]),end='')
    print("SC: {:.5f}".format(metric_SC[name]))
    
    clusters = pd.DataFrame(cluster_predict[name],index=X.index,columns=['cluster'])
    X_cluster = pd.concat([X,clusters],axis=1)
    min_size = 5
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]

    makeScatterPlot(X_filtrado)
    makeHeatmap(X_filtrado)
    
    
clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
X_cluster = pd.concat([X,clusters],axis=1)

min_size = 5
X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
k_filtrado = len(set(X_filtrado['cluster']))

X_filtrado = X_filtrado.drop('cluster',1)
X_filtrado_normal = X_filtrado.apply(norm_to_zero_one)

linkage_array = hierarchy.ward(X_filtrado_normal)
h_dict = hierarchy.dendrogram(linkage_array,orientation='left')

sns.clustermap(X_filtrado_normal, method='ward', col_cluster=False, figsize=(15,7), cmap='YlGnBu', yticklabels=False)