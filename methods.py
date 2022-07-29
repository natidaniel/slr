#!/usr/bin/env python
# coding: utf-8

import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
#import math as ma

#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score
#import seaborn as sns

#from collections import Counter|
#from sklearn.ensemble import IsolationForest
#from sklearn import svm

from pyod.models.knn import KNN

# from keras.models import load_model
from tensorflow.python.keras.models import Model, load_model

from sklearn import preprocessing
from sklearn.manifold import TSNE

#import pandas as pd
#import pylab
import keras
import tensorflow.compat.v1 as tf


os.environ['CUDA_VISIBLE_DEVICES'] = str('0')

mode   = ['devicemode']
SAMPLE_FREQ = 50 
FILE_MARGINES = 2*SAMPLE_FREQ
MODE_LABELS = ['pocket','swing','texting','talking', 'Unknown']


def SignleLoader(root,file):
    data=pd.read_csv(os.path.join(root,file))
    if len(data) < 400 :
        print (' only ' , len(data) , ' samples in file ', file , ' pass ')
        return pd.DataFrame() 
    if 'p' in data.columns: # remove barometer data if exsits
        data = data.drop(['p'], axis=1)
    if 'gfx' in data.columns: # fix bug in recording data
        data.rename(columns={'gfx': 'gFx'}, inplace=True) 
    data = data.drop(data.columns[len(data.columns)-1], axis=1) # delete unrelevent last column
    data['source']=file  
    data['SmartphoneMode']=MODE_LABELS[-1] ## 'whatever' label 
    data['devicemode'] = len(MODE_LABELS) #-1
        ## search device mode label in file name and add as new properties :
    for label in MODE_LABELS:
        if label.lower() in file.lower():  
           data['SmartphoneMode']=label         ## label name 
           data['devicemode'] = MODE_LABELS.index(label)    ## label index 
           break
        ## crop samples from start and from the end of the file :
    margin = min(len(data) / 2 - 1 , FILE_MARGINES)
    data.drop(data.index[range(0,margin)],axis=0,inplace=True)
    data.drop(data.index[range(-margin,-1)],axis=0,inplace=True)
    # verify norm
    data_acc = preprocessing.normalize(data.iloc[:,[1,2,3]], norm='l2')
    data_acc_DF = pd.DataFrame(data_acc)
    data['gFx'] = data_acc_DF[0].values
    data['gFy'] = data_acc_DF[1].values
    data['gFz'] = data_acc_DF[2].values
    
    # add features
    data['gSTD'] = np.std([data['gFx'],data['gFy'],data['gFz']],axis=0)
#    data['theta'] =np.arctan(data['gFx']/np.sqrt(data['gFy']**2+data['gFz']**2))
#    data['roll']= np.arctan2(-data['gFy'],-data['gFz'])
    data['gmul']= data['gFx']*data['gFy']*data['gFz']
    
    data['wMag'] = np.sqrt(data['wx']**2+data['wy']**2+data['wz']**2)
    data['wSTD'] = np.std([data['wx'],data['wy'],data['wz']],axis=0)
    data['wDiff'] = 0 - data['wMag']
    data['wmul']=data['wx']*data['wy']*data['wz']

    data['gwSTD'] = data['gSTD']*data['wSTD']
    data['gwmul'] = data['gmul']*data['wmul']
    
    print('loading : ' , file) 
    print('loading : ' , len(data) , ' samples from ', file)  
    return data 

def FilesLoader(filesLoc):
    print ('Loading files from: ' , filesLoc )
    return pd.concat([SignleLoader(filesLoc,f) for f in os.listdir(filesLoc) if f.lower().endswith('.csv')])


def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def toLstmFormat(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:,1:7])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5) # 5 is number of modes
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y

def toLstmFormatAcc(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:,1:4])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5) # 5 is number of modes
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y

def toLstmFormatWithFeatures(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:,[1,2,3,4,5,6,14,15,16,17,18,19,20,21]])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5) # 5 is number of modes
    x = np.array         ([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)]) 
    y = ydata[:len(x)]
    return x, y

def toLstmFormatWithFeaturesAcc(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:,[1,2,3,14,15]])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5) # 5 is number of modes
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y

def load_into_x_y(dirname, TimeStep=32):
    data = FilesLoader(dirname)
    x, y = toLstmFormatAcc(data,TimeStep)
    return x,y

def plot_umap(X_embed, y_plot):
    colors = ['r', 'g', 'b', 'y','m']
    plt.figure(figsize=(12,12))
    for i,c in enumerate(colors):
        plt.plot(X_embed[y_plot==i,0], X_embed[y_plot==i,1], c+'.')
    plt.show()


#################################################################

def TicTocGenerator():
    """Generator that returns time differences."""
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti

TicToc = TicTocGenerator()


def toc(tempBool=True):
    """Prints the time difference yielded by generator instance TicToc."""
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )


def tic():
    """Records a time in TicToc, marks the beginning of a time interval."""
    toc(False)


#################################################################
# load files
testSourceMoreTest = "/home/maintenance/projects/Unseen Mode 2/Data/Test_OBW_15_2"
trainSourceMoreTest = "/home/maintenance/projects/Unseen Mode 2/Data/Train"
x_train, y_train = load_into_x_y(trainSourceMoreTest)
x_test, y_test = load_into_x_y(testSourceMoreTest)

###################################################################
# choose trained model
model = load_model("smartphone_model_cnnAcc_2.hdf5")
#model = load_model("smartphone_model3_256Acc.hdf5")
model.summary()

# baseline
print('Start baseline')
b_thes = 0.95
baseline_test = model.predict(x_test)
print(baseline_test.shape)
y_test_int = np.argmax(y_test, axis=1)
Unknown_baseline = np.ones(len(y_test_int))  # default is unknown
for i in range(len(y_test_int)):
    is_uknown = 1  # default is unknown
    if (baseline_test[i][0] > b_thes) or (baseline_test[i][1] > b_thes) or (baseline_test[i][2] > b_thes) or (baseline_test[i][3] > b_thes):
        is_uknown = 0  # otherwise we found it is strongly connected to other state
    Unknown_baseline[i] = is_uknown  # update the

correctly_detected_unknowns = np.sum(Unknown_baseline == 1)
print('baseline:', correctly_detected_unknowns, correctly_detected_unknowns / len(y_test_int))
print('END baseline')

#layer_name = 'dense_5'
layer_name = 'dense_1'

# method
isKNN = True
isPCA = False
isLDA = False

# create new model without last layer
layer_output=model.get_layer(layer_name).output
intermediate_model=Model(inputs=model.input,outputs=layer_output)

dense_train = intermediate_model.predict(x_train)
dense_test = intermediate_model.predict(x_test)

print(dense_train.shape)
print(dense_test.shape)

y_train_int = np.argmax(y_train, axis=1)
y_test_int = np.argmax(y_test, axis=1)

if isKNN:
    tic()
    # k=5
    clf_name = 'KNN'
    clf = KNN()
    clf.fit(dense_train)
    y_test_pred = clf.predict(dense_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(dense_test)  # outlier scores
    correctly_detected_unknowns = np.sum(y_test_pred == 1)
    print('knn only:', correctly_detected_unknowns, correctly_detected_unknowns / len(y_test_pred))
    toc()

if isPCA:
    # PCA
    tic()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    print(dense_train.shape)
    W = pca.fit(dense_train)

    X_train = pca.transform(dense_train)
    X_test = pca.transform(dense_test)
    print('shapes:', X_train.shape, X_test.shape)
    print(y_train_int.shape, y_test_int.shape)

    clf_name = 'KNN'
    clf = KNN()
    clf.fit(X_train)

    # PCA
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores
    correctly_detected_unknowns = np.sum(y_test_pred == 1)
    print('pca:',correctly_detected_unknowns, correctly_detected_unknowns/len(y_test_pred))
    toc()

if isLDA:
    # LDA
    tic()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(dense_train, y_train_int)

    X_train_lda = lda.transform(dense_train)
    X_test_lda = lda.transform(dense_test)
    print('shapes:', X_train_lda.shape, X_test_lda.shape)
    print(y_train_int.shape, y_test_int.shape)

    clf_name = 'KNN'
    clf = KNN()
    clf.fit(X_train_lda)
    y_test_lda_pred = clf.predict(X_test_lda)  # outlier labels (0 or 1)
    y_test_lda_scores = clf.decision_function(X_test_lda)  # outlier scores
    correctly_detected_unknowns_lda = np.sum(y_test_lda_pred == 1)
    print('lda:',correctly_detected_unknowns_lda, correctly_detected_unknowns_lda/len(y_test_lda_pred))
    toc()


def plot_sk_pca(X, Y):
    # plot pca
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(X[Y,0], X[Y, 1], color='r', marker='*', label='Known')
    ax.scatter(X[~Y,0], X[~Y, 1], color='b', marker='x', label='UnKnown')
    ax.grid()
    ax.legend()
    ax.set_title("2D PCA UnKnown Dataset")


# Kernel PCA - Nati
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons


def plot_pca_circles(X, Y, SAMPLE_FREQ):
    X = X[0::SAMPLE_FREQ]
    y = Y[0::SAMPLE_FREQ]
    pca = PCA(n_components=2)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)  # Radial Basis Function kernel
    X_pca = pca.fit_transform(X)
    X_kpca = kpca.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(10,4))

    # original data
    ax1 = fig.add_subplot(1,3,1)
    ax1.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    ax1.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax1.grid()
    ax1.set_title("original data")

    # pca
    ax2 = fig.add_subplot(1,3,2)
    ax2.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax2.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax2.grid()
    ax2.set_title("PCA")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # kpca
    ax3 = fig.add_subplot(1,3,3)
    ax3.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax3.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax3.grid()
    ax3.set_title("KPCA")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")

    plt.tight_layout()


def plot_pca_comp(X, Y, SAMPLE_FREQ, num_c, p_x, p_y):
    X = X[0::SAMPLE_FREQ]
    y = Y[0::SAMPLE_FREQ]
    pca = PCA(n_components=num_c)
    X_pca = pca.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(8, 8))

    # pca
    if num_c == 2:
        ax = fig.add_subplot(1,1,1)
        ax.scatter(X_pca[y==p_x, 0], X_pca[y==p_x, 1], color='red', marker='^', alpha=0.5)
        ax.scatter(X_pca[y==p_y, 0], X_pca[y==p_y, 1], color='blue', marker='o', alpha=0.5)
        ax.grid()
        ax.set_title("PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', marker='^', alpha=0.5)
        ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', marker='o', alpha=0.5)
        ax.scatter(X_pca[y == 3, 0], X_pca[y == 3, 1], color='black', marker='*', alpha=0.5)
        ax.grid()
        ax.set_title("PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    plt.tight_layout()

plot_pca_circles(dense_train, y_train_int, 24)
plot_pca_circles(dense_test, y_test_int, 24)


def plot_pca_moons(X, Y, SAMPLE_FREQ):
    # example - moons
    X = X[0::SAMPLE_FREQ]
    y = Y[0::SAMPLE_FREQ]
    pca = PCA(n_components=2)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)  # Radial Basis Function kernel
    X_pca = pca.fit_transform(X)
    X_kpca = kpca.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(10,4))

    # original data
    ax1 = fig.add_subplot(1,3,1)
    ax1.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    ax1.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax1.grid()
    ax1.set_title("original data")

    # pca
    ax2 = fig.add_subplot(1,3,2)
    ax2.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax2.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax2.grid()
    ax2.set_title("PCA")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # kpca
    ax3 = fig.add_subplot(1,3,3)
    ax3.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax3.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax3.grid()
    ax3.set_title("KPCA")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")

    plt.tight_layout()


def plot_pca_moons2(X, Y, SAMPLE_FREQ):
    # example - moons
    X = X[0::SAMPLE_FREQ]
    y = Y[0::SAMPLE_FREQ]
    pca = PCA(n_components=2)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)  # Radial Basis Function kernel
    X_pca = pca.fit_transform(X)
    X_kpca = kpca.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(10,4))

    # pca
    ax2 = fig.add_subplot(1,2,1)
    ax2.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax2.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax2.grid()
    ax2.set_title("PCA")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # kpca
    ax3 = fig.add_subplot(1,2,2)
    ax3.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax3.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax3.grid()
    ax3.set_title("KPCA")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    plt.tight_layout()


plot_pca_moons2(dense_train, y_train_int, 50)
plot_pca_moons(dense_train, y_train_int, 24)
plot_pca_moons(dense_test, y_test_int, 24)


# t-SNE Nati
from tsne import perform_tsne
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def plot_tsne(X, y, dim=2, perplexity=30.0, scale_data=False):
    if dim < 2 or dim > 3:
        print("OH NO :(")
        raise SystemError("2 <= dim <= 3")
    t_sne = TSNE(n_components=dim, perplexity=perplexity)
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    X_embedded = t_sne.fit_transform(X)
    if dim == 2:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], color='red', marker='^', alpha=0.5)
        ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], color='blue', marker='o', alpha=0.5)
        ax.grid()
        ax.legend()
        ax.set_title("2D t-SNE")
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], X_embedded[y == 0, 2], color='r', marker='*', label='Known mode')
        ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], X_embedded[y == 1, 2], color='b', marker='x', label='Unknown mode')
        ax.grid()
        ax.legend()
        ax.set_title("3D t-SNE")


X = dense_train[0::SAMPLE_FREQ]
y = y_train_int[0::SAMPLE_FREQ]
# let's try to reduce to 2 dimensions, with scaling
plot_tsne(X, y, dim=2, perplexity=10.0, scale_data=True)

# let's try to reduce to 2 dimensions, without scaling
plot_tsne(X, y, dim=2, perplexity=30.0, scale_data=False)

# let's try to reduce to 3 dimensions
plot_tsne(X, y, dim=3, perplexity=30.0, scale_data=True)

# other longer error version
perform_tsne(X_data=dense_train,y_data=y_train_int, perplexities =[20])
perform_tsne(X_data=dense_train,y_data=y_train_int, perplexities =[2,50])
perform_tsne(X_data=dense_test,y_data=y_test_int, perplexities =[2,20,50])