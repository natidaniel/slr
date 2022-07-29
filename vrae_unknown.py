from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from random import randint
from plotly.graph_objects import Scatter, Data, Layout, Figure
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch import distributions
from torch.autograd import Variable
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd
import argparse
import torch
import os
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode()


import time


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


SampleOut = 100


# Kernel PCA - Nati
def plot_pca_moons(X, y):
    # example - moons
    pca = PCA(n_components=2)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)  # Radial Basis Function kernel
    X_pca = pca.fit_transform(X)
    X_kpca = kpca.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(10,4))

    # original data
    ax1 = fig.add_subplot(1,3,1)
    sr = 30
    # ax1.scatter(X[y==0, 0][1:sr], X[y==0, 1][1:sr], color='red', marker='*', alpha=0.5, label='handheld')
    # ax1.scatter(X[y==1, 0][1:sr], X[y==1, 1][1:sr], color='blue', marker='o', alpha=0.5, label='swing')
    # ax1.scatter(X[y == 2, 0][1:sr], X[y == 2, 1][1:sr], color='green', marker='^', alpha=0.5, label='texting')
    # ax1.scatter(X[y == 3, 0][1:sr], X[y == 3, 1][1:sr], color='black', marker='x', alpha=0.5, label='scanning')
    ax1.scatter(X[y==0, 0][1:sr], X[y==0, 1][1:sr], color='red', marker='*', alpha=0.5, label='pocket')
    ax1.scatter(X[y==1, 0][1:sr], X[y==1, 1][1:sr], color='blue', marker='o', alpha=0.5, label='swing')
    ax1.scatter(X[y == 2, 0][1:sr], X[y == 2, 1][1:sr], color='green', marker='^', alpha=0.5, label='texting')
    ax1.scatter(X[y == 3, 0][1:sr], X[y == 3, 1][1:sr], color='black', marker='x', alpha=0.5, label='talking')
    #ax1.scatter(X[y == 4, 0][1:sr], X[y == 4, 1][1:sr], color='purple', marker='+', alpha=0.5, label='unknown')
    ax1.grid()
    ax1.legend()
    ax1.set_title("original data")

    # pca
    ax2 = fig.add_subplot(1,3,2)
    # ax2.scatter(X_pca[y==0, 0][1:sr], X_pca[y==0, 1][1:sr], color='red', marker='*', alpha=0.5, label='handheld')
    # ax2.scatter(X_pca[y==1, 0][1:sr], X_pca[y==1, 1][1:sr], color='blue', marker='o', alpha=0.5, label='swing')
    # ax2.scatter(X_pca[y==2, 0][1:sr], X_pca[y==2, 1][1:sr], color='green', marker='^', alpha=0.5, label='texting')
    # ax2.scatter(X_pca[y==3, 0][1:sr], X_pca[y==3, 1][1:sr], color='black', marker='x', alpha=0.5, label='scanning')
    ax2.scatter(X_pca[y==0, 0][1:sr], X_pca[y==0, 1][1:sr], color='red', marker='*', alpha=0.5, label='pocket')
    ax2.scatter(X_pca[y==1, 0][1:sr], X_pca[y==1, 1][1:sr], color='blue', marker='o', alpha=0.5, label='swing')
    ax2.scatter(X_pca[y==2, 0][1:sr], X_pca[y==2, 1][1:sr], color='green', marker='^', alpha=0.5, label='texting')
    ax2.scatter(X_pca[y==3, 0][1:sr], X_pca[y==3, 1][1:sr], color='black', marker='x', alpha=0.5, label='talking')
    #ax2.scatter(X_pca[y == 4, 0][1:sr], X_pca[y == 4, 1][1:sr], color='purple', marker='+', alpha=0.5, label='unknown')
    ax2.grid()
    ax2.set_title("PCA")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()

    # kpca
    ax3 = fig.add_subplot(1,3,3)
    # ax3.scatter(X_kpca[y==0, 0][1:sr], X_kpca[y==0, 1][1:sr], color='red', marker='*', alpha=0.5, label='handheld')
    # ax3.scatter(X_kpca[y==1, 0][1:sr], X_kpca[y==1, 1][1:sr], color='blue', marker='o', alpha=0.5, label='swing')
    # ax3.scatter(X_kpca[y==2, 0][1:sr], X_kpca[y==2, 1][1:sr], color='green', marker='^', alpha=0.5, label='texting')
    # ax3.scatter(X_kpca[y==3, 0][1:sr], X_kpca[y==3, 1][1:sr], color='black', marker='x', alpha=0.5, label='scanning')
    ax3.scatter(X_kpca[y==0, 0][1:sr], X_kpca[y==0, 1][1:sr], color='red', marker='*', alpha=0.5, label='pocket')
    ax3.scatter(X_kpca[y==1, 0][1:sr], X_kpca[y==1, 1][1:sr], color='blue', marker='o', alpha=0.5, label='swing')
    ax3.scatter(X_kpca[y==2, 0][1:sr], X_kpca[y==2, 1][1:sr], color='green', marker='^', alpha=0.5, label='texting')
    ax3.scatter(X_kpca[y==3, 0][1:sr], X_kpca[y==3, 1][1:sr], color='black', marker='x', alpha=0.5, label='talking')
    #ax3.scatter(X_kpca[y == 4, 0][1:sr], X_kpca[y == 4, 1][1:sr], color='purple', marker='+', alpha=0.5,label='unknown')
    ax3.grid()
    ax3.set_title("KPCA")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.legend()

    plt.tight_layout()


def plot_pca_moons2(X, y):
    # example - moons
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


def plot_pca_circles2(X, y):
    pca = PCA(n_components=2)
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)  # Radial Basis Function kernel
    X_pca = pca.fit_transform(X)
    X_kpca = kpca.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(10,4))

    # original data
    ax1 = fig.add_subplot(1,3,1)
    ax1.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='^', alpha=0.5)
    ax1.scatter(X[y==3, 0], X[y==3, 1], color='blue', marker='o', alpha=0.5)
    ax1.grid()
    ax1.set_title("original data")

    # pca
    ax2 = fig.add_subplot(1,3,2)
    ax2.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='red', marker='^', alpha=0.5)
    ax2.scatter(X_pca[y==3, 0], X_pca[y==3, 1], color='blue', marker='o', alpha=0.5)
    ax2.grid()
    ax2.set_title("PCA")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # kpca
    ax3 = fig.add_subplot(1,3,3)
    ax3.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='red', marker='^', alpha=0.5)
    ax3.scatter(X_kpca[y==3, 0], X_kpca[y==3, 1], color='blue', marker='o', alpha=0.5)
    ax3.grid()
    ax3.set_title("KPCA")
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")

    plt.tight_layout()

# tsne tain+val
def plot_tsne_plus_val(X, y, dim=2, perplexity=30.0, scale_data=False):
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
        plt.rcParams["axes.grid"] = False
        ax = fig.add_subplot(1,1,1)
        se = 50
        ax.scatter(X_embedded[y == 0, 0][1:se], X_embedded[y == 0, 1][1:se], color='red', marker='*', alpha=0.5, label='pocket')
        ax.scatter(X_embedded[y == 1, 0][1:se], X_embedded[y == 1, 1][1:se], color='blue', marker='x', alpha=0.5, label='swing')
        ax.scatter(X_embedded[y == 2, 0][1:se], X_embedded[y == 2, 1][1:se], color='green', marker='^', alpha=0.5, label='texting')
        ax.scatter(X_embedded[y == 3, 0][1:se], X_embedded[y == 3, 1][1:se], color='black', marker='o', alpha=0.5, label='talking')
        ax.scatter(X_embedded[y == 4, 0][1:se], X_embedded[y == 4, 1][1:se], color='purple', marker='+', alpha=0.5, label='Unknown')
        #ax.grid(None)
        ax.legend()
        ax.set_xlabel('tsne1', fontsize=14)
        ax.set_ylabel('tsne2', fontsize=14)
        ax.set_title("2D t-SNE")
    else:
        fig = plt.figure(figsize=(8, 8))
        plt.rcParams["axes.grid"] = False
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        se = 50
        ax.scatter(X_embedded[y == 0, 0][1:se], X_embedded[y == 0, 1][1:se], X_embedded[y == 0, 2][1:se], color='r', marker='*', label='pocket')
        ax.scatter(X_embedded[y == 1, 0][1:se], X_embedded[y == 1, 1][1:se], X_embedded[y == 1, 2][1:se], color='b', marker='x', label='swing')
        ax.scatter(X_embedded[y == 2, 0][1:se], X_embedded[y == 2, 1][1:se], X_embedded[y == 2, 2][1:se], color='g', marker='^',label='texting')
        ax.scatter(X_embedded[y == 3, 0][1:se], X_embedded[y == 3, 1][1:se], X_embedded[y == 3, 2][1:se], color='k', marker='o', label='talking')
        ax.scatter(X_embedded[y == 4, 0][1:se], X_embedded[y == 4, 1][1:se], X_embedded[y == 4, 2][1:se], color='c', marker='+', label='Unknown')
        ax.grid(None)
        ax.legend()
        ax.set_xlabel('tsne1', fontsize=14)
        ax.set_ylabel('tsne2', fontsize=14)
        ax.set_zlabel('tsne3', fontsize=14)
        plt.savefig('3D t-SNE.png', bbox_inches='tight', dpi=32)
        ax.set_title("3D t-SNE")

# t-SNE Nati
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
        plt.rcParams["axes.grid"] = False
        ax = fig.add_subplot(1,1,1)
        se = 50
        # ax.scatter(X_embedded[y == 0, 0][1:se], X_embedded[y == 0, 1][1:se], color='red', marker='*', alpha=0.5, label='handheld')
        # ax.scatter(X_embedded[y == 1, 0][1:se], X_embedded[y == 1, 1][1:se], color='blue', marker='x', alpha=0.5, label='swing')
        # ax.scatter(X_embedded[y == 2, 0][1:se], X_embedded[y == 2, 1][1:se], color='green', marker='^', alpha=0.5, label='texting')
        # ax.scatter(X_embedded[y == 3, 0][1:se], X_embedded[y == 3, 1][1:se], color='black', marker='o', alpha=0.5, label='scanning')
        ax.scatter(X_embedded[y == 0, 0][1:se], X_embedded[y == 0, 1][1:se], color='red', marker='*', alpha=0.5, label='pocket')
        ax.scatter(X_embedded[y == 1, 0][1:se], X_embedded[y == 1, 1][1:se], color='blue', marker='x', alpha=0.5, label='swing')
        ax.scatter(X_embedded[y == 2, 0][1:se], X_embedded[y == 2, 1][1:se], color='green', marker='^', alpha=0.5, label='texting')
        ax.scatter(X_embedded[y == 3, 0][1:se], X_embedded[y == 3, 1][1:se], color='black', marker='o', alpha=0.5, label='talking')
        #ax.scatter(X_embedded[y == 4, 0][1:se], X_embedded[y == 4, 1][1:se], color='purple', marker='+', alpha=0.5,
        #           label='Unknown')
        #ax.grid()
        ax.grid(None)
        ax.legend()
        ax.set_xlabel('tsne1', fontsize=14)
        ax.set_ylabel('tsne2', fontsize=14)
        ax.set_title("2D t-SNE")
    else:
        fig = plt.figure(figsize=(8, 8))
        plt.rcParams["axes.grid"] = False
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        se = 50
        # ax.scatter(X_embedded[y == 0, 0][1:se], X_embedded[y == 0, 1][1:se], X_embedded[y == 0, 2][1:se], color='r', marker='*', label='handheld')
        # ax.scatter(X_embedded[y == 1, 0][1:se], X_embedded[y == 1, 1][1:se], X_embedded[y == 1, 2][1:se], color='b', marker='x', label='swing')
        # ax.scatter(X_embedded[y == 2, 0][1:se], X_embedded[y == 2, 1][1:se], X_embedded[y == 2, 2][1:se], color='g', marker='^',label='texting')
        # ax.scatter(X_embedded[y == 3, 0][1:se], X_embedded[y == 3, 1][1:se], X_embedded[y == 3, 2][1:se], color='k', marker='o',
        #            label='scanning')
        ax.scatter(X_embedded[y == 0, 0][1:se], X_embedded[y == 0, 1][1:se], X_embedded[y == 0, 2][1:se], color='r', marker='*', label='pocket')
        ax.scatter(X_embedded[y == 1, 0][1:se], X_embedded[y == 1, 1][1:se], X_embedded[y == 1, 2][1:se], color='b', marker='x', label='swing')
        ax.scatter(X_embedded[y == 2, 0][1:se], X_embedded[y == 2, 1][1:se], X_embedded[y == 2, 2][1:se], color='g', marker='^',label='texting')
        ax.scatter(X_embedded[y == 3, 0][1:se], X_embedded[y == 3, 1][1:se], X_embedded[y == 3, 2][1:se], color='k', marker='o', label='talking')
        #ax.scatter(X_embedded[y == 4, 0][1:se], X_embedded[y == 4, 1][1:se], X_embedded[y == 4, 2][1:se], color='c', marker='+', label='Unknown')
        ax.grid(None)
        ax.legend()
        ax.set_xlabel('tsne1', fontsize=14)
        ax.set_ylabel('tsne2', fontsize=14)
        ax.set_zlabel('tsne3', fontsize=14)
        plt.savefig('3D t-SNE.png', bbox_inches='tight', dpi=32)
        ax.set_title("3D t-SNE")


def plot_clustering(z_run, labels, engine ='matplotlib', download = True, folder_name ='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """
    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(z_run)

        trace = Scatter(
            x=z_run_pca[:, 0],
            y=z_run_pca[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='PCA on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(
            x=z_run_tsne[:, 0],
            y=z_run_tsne[:, 1],
            mode='markers',
            marker=dict(color=colors)
        )
        data = Data([trace])
        layout = Layout(
            title='tSNE on z_run',
            showlegend=False
        )
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name):

        labels = labels[:z_run.shape[0]] # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)

        plt.scatter(z_run_pca[:, 0], z_run_pca[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('PCA on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/pca.png")
        else:
            plt.show()

        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=300).fit_transform(z_run)
        plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
        plt.title('tSNE on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/tsne.png")
        else:
            plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)


def open_data(direc, ratio_train=0.8):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]

# scanning and handheld
def load_into_x_y(dirname, TimeStep=50):
    data = FilesLoader(dirname)
    x, y = toLstmFormatAcc(data, TimeStep)
    return x, y

def FilesLoader(filesLoc):
    print('Loading files from: ', filesLoc)
    return pd.concat([SignleLoader(filesLoc, f) for f in os.listdir(filesLoc) if f.lower().endswith('.csv')])


def SignleLoader(root, file):  # read each file in dir
    data = pd.read_csv(os.path.join(root, file))
    UserLabels = ['handheld', 'swing', 'texting', 'unknown']  # unknow == scanning
    if np.size(data, axis=1) == 1:  # this is for old slam files
        data = pd.read_csv(os.path.join(root, file), sep=r'\t+')
    print(file)
    if len(data) < 3 * SampleOut:  # dont use small size files
        print(' only ', len(data), ' samples in file ', file, ' pass ')
        return pd.DataFrame()
    data['source'] = file
    data['SmartphoneMode'] = UserLabels[-1]  ## 'other' label
    data['UserMode'] = len(UserLabels) - 1
    for label in UserLabels:
        if label.lower() in file.lower():
            data['SmartphoneMode'] = label  ## label name
            data['UserMode'] = UserLabels.index(label)  ## label index
            break
    if np.size(data, axis=1) == 16:
        return pd.DataFrame()
    if np.size(data, axis=1) == 10:  # read SLAM files
        data.insert(loc=1, column='time1', value=data.iloc[:, 0])  # dummy columns
        data.columns = ['sampleTime', 'boottime', 'accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'source',
                        'SmartphoneMode', 'UserMode']
        data.loc[:, 'magx'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'magy'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'magz'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'pitch'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'roll'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'azimuth'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvectx'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvecty'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvectz'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'pressure'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'stepcounter'] = (np.zeros(len(data['gyrox'])))
        listNew1 = ['sampleTime', 'boottime', 'gyrox', 'gyroy', 'gyroz', 'accx', 'accy', 'accz', 'magx', 'magy', 'magz',
                    'pitch', 'roll', 'azimuth', 'rotationvectx', 'rotationvecty', 'rotationvectz', 'pressure',
                    'stepcounter', 'source', 'SmartphoneMode', 'UserMode']
        data = data[listNew1]  # rearrange columns

    #        data = data[data.index % 2 != 0]

    if np.size(data, axis=1) == 15:  # read my files
        data = data.drop(data.columns[7:12], axis=1)  # delete unrelevent  column
        data.insert(loc=1, column='time1', value=data.iloc[:, 0])  # dummy columns
        clistNew = ['time', 'time1', 'wx', 'wy', 'wz', 'gFx', 'gFy', 'gFz', 'source', 'SmartphoneMode', 'UserMode']
        data = data[clistNew]  # rearrange columns
        data.columns = ['sampleTime', 'boottime', 'gyrox', 'gyroy', 'gyroz', 'accx', 'accy', 'accz', 'source',
                        'SmartphoneMode', 'UserMode']
        data.loc[:, 'magx'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'magy'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'magz'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'pitch'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'roll'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'azimuth'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvectx'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvecty'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvectz'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'pressure'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'stepcounter'] = (np.zeros(len(data['gyrox'])))
        listNew1 = ['sampleTime', 'boottime', 'gyrox', 'gyroy', 'gyroz', 'accx', 'accy', 'accz', 'magx', 'magy', 'magz',
                    'pitch', 'roll', 'azimuth', 'rotationvectx', 'rotationvecty', 'rotationvectz', 'pressure',
                    'stepcounter', 'source', 'SmartphoneMode', 'UserMode']
        data = data[listNew1]  # rearrange columns
        data['accx'] = data['accx'] * 9.81  # from g-->m/s^2
        data['accy'] = data['accy'] * 9.81  # from g-->m/s^2
        data['accz'] = data['accz'] * 9.81  # from g-->m/s^2

    if np.size(data, axis=1) == 14:  # read RIDI files
        data = data.drop(data.columns[7:11], axis=1)  # delete unrelevent  column
        data.insert(loc=1, column='time1', value=data.iloc[:, 0])  # dummy columns
        clistNew = ['time', 'time1', 'wx', 'wy', 'wz', 'gFx', 'gFy', 'gFz', 'source', 'SmartphoneMode', 'UserMode']
        data = data[clistNew]  # rearrange columns
        data.columns = ['sampleTime', 'boottime', 'gyrox', 'gyroy', 'gyroz', 'accx', 'accy', 'accz', 'source',
                        'SmartphoneMode', 'UserMode']
        data.loc[:, 'magx'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'magy'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'magz'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'pitch'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'roll'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'azimuth'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvectx'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvecty'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'rotationvectz'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'pressure'] = (np.zeros(len(data['gyrox'])))
        data.loc[:, 'stepcounter'] = (np.zeros(len(data['gyrox'])))
        listNew1 = ['sampleTime', 'boottime', 'gyrox', 'gyroy', 'gyroz', 'accx', 'accy', 'accz', 'magx', 'magy', 'magz',
                    'pitch', 'roll', 'azimuth', 'rotationvectx', 'rotationvecty', 'rotationvectz', 'pressure',
                    'stepcounter', 'source', 'SmartphoneMode', 'UserMode']
        data = data[listNew1]  # rearrange columns
    if np.size(data, axis=1) == 23:  # read DanDans files
        data = data.drop(data.columns[2], axis=1)  # delete unrelevent last column
        data['gyrox'] = data['gyrox'] * np.pi / 180
        data['gyroy'] = data['gyroy'] * np.pi / 180
        data['gyroz'] = data['gyroz'] * np.pi / 180
    Acc = data.iloc[:, [5, 6, 7]]  # m/s^2
    AngularRate = data.iloc[:, [2, 3, 4]]  # rad/sec
    data['AccNorm'] = np.sqrt(Acc.iloc[:, 0] ** 2 + Acc.iloc[:, 1] ** 2 + Acc.iloc[:, 2] ** 2)
    data['AngularRateNorm'] = np.sqrt(
        AngularRate.iloc[:, 0] ** 2 + AngularRate.iloc[:, 1] ** 2 + AngularRate.iloc[:, 2] ** 2)
    data['max'] = data[["accx", "accy", "accz"]].max(axis=1) * data[["gyrox", "gyroy", "gyroz"]].max(axis=1)
    data['min'] = data[["accx", "accy", "accz"]].min(axis=1) * data[["gyrox", "gyroy", "gyroz"]].min(axis=1)
    data['max_acc'] = data[["accx", "accy", "accz"]].max(axis=1)
    data['min_acc'] = data[["accx", "accy", "accz"]].min(axis=1)
    data['max_gyro'] = data[["gyrox", "gyroy", "gyroz"]].max(axis=1)
    data['min_gyro'] = data[["gyrox", "gyroy", "gyroz"]].min(axis=1)
    data['std_acc'] = np.std(Acc, axis=1)
    data['mean_acc'] = np.mean(Acc, axis=1)
    data['abs_accx'] = np.abs(data['accx'])
    data['abs_accy'] = np.abs(data['accy'])
    data['abs_accz'] = np.abs(data['accz'])
    data['std_gyro'] = np.std(AngularRate, axis=1)
    data['std_ag'] = data['std_gyro'] * data['std_acc']

    margin = min(len(data) / 2 - 1, SampleOut)

    data.drop(data.index[range(0, margin)], axis=0, inplace=True)
    data.drop(data.index[range(-margin, -1)], axis=0, inplace=True)
    print('loading : ', file)
    print('loading : ', len(data), ' samples from ', file)
    return data

def toLstmFormatAcc(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:, 2:8])
    ydata = np.asarray(data[['UserMode']])
    #ydata = indices_to_one_hot(ydata, 4)
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y


def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def load_into_x_y_old_data(dirname, TimeStep=32):
    data = FilesLoader_old_data(dirname)
    x, y = toLstmFormatAcc_old_data(data, TimeStep)
    return x, y

def FilesLoader_old_data(filesLoc):
    print('Loading files from: ', filesLoc)
    return pd.concat([SignleLoader_old_data(filesLoc, f) for f in os.listdir(filesLoc) if f.lower().endswith('.csv')])


def toLstmFormatAcc_old_data(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:, 1:4])
    ydata = np.asarray(data[mode])
    #ydata = indices_to_one_hot(ydata, 5)
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y


def SignleLoader_old_data(root, file):
    data = pd.read_csv(os.path.join(root, file))
    if len(data) < 400:
        print(' only ', len(data), ' samples in file ', file, ' pass ')
        return pd.DataFrame()
    if 'p' in data.columns:  # remove barometer data if exsits
        data = data.drop(['p'], axis=1)
    if 'gfx' in data.columns:  # fix bug in recording data
        data.rename(columns={'gfx': 'gFx'}, inplace=True)

    # relevant only to new data
    #data = data.drop(data.columns[len(data.columns) - 1], axis=1)  # delete unrelevent last column

    data['source'] = file
    data['SmartphoneMode'] = MODE_LABELS[-1]  ## 'whatever' label
    data['devicemode'] = len(MODE_LABELS)

    ## search device mode label in file name and add as new properties :
    for label in MODE_LABELS:
        if label.lower() in file.lower():
            data['SmartphoneMode'] = label  ## label name
            data['devicemode'] = MODE_LABELS.index(label)  ## label index
            break
        ## crop samples from start and from the end of the file :
    margin = min(len(data) / 2 - 1, FILE_MARGINES)
    data.drop(data.index[range(0, margin)], axis=0, inplace=True)
    data.drop(data.index[range(-margin, -1)], axis=0, inplace=True)
    # verify norm
    data_acc = preprocessing.normalize(data.iloc[:, [1, 2, 3]], norm='l2')
    data_acc_DF = pd.DataFrame(data_acc)
    data['gFx'] = data_acc_DF[0].values
    data['gFy'] = data_acc_DF[1].values
    data['gFz'] = data_acc_DF[2].values

    # add features
    data['gSTD'] = np.std([data['gFx'], data['gFy'], data['gFz']], axis=0)
    #    data['theta'] =np.arctan(data['gFx']/np.sqrt(data['gFy']**2+data['gFz']**2))
    #    data['roll']= np.arctan2(-data['gFy'],-data['gFz'])
    data['gmul'] = data['gFx'] * data['gFy'] * data['gFz']

    data['wMag'] = np.sqrt(data['wx'] ** 2 + data['wy'] ** 2 + data['wz'] ** 2)
    data['wSTD'] = np.std([data['wx'], data['wy'], data['wz']], axis=0)
    data['wDiff'] = 0 - data['wMag']
    data['wmul'] = data['wx'] * data['wy'] * data['wz']

    data['gwSTD'] = data['gSTD'] * data['wSTD']
    data['gwmul'] = data['gmul'] * data['wmul']

    print('loading : ', file)
    print('loading : ', len(data), ' samples from ', file)
    return data


# configurations
#MODE_LABELS = ['pocket','swing','texting','talking','Unknown']
MODE_LABELS = ['pocket','swing','texting','talking']
mode = ['devicemode']
SAMPLE_FREQ = 50
FILE_MARGINES = 2*SAMPLE_FREQ


class BaseEstimator(SklearnBaseEstimator):

    def summarize(self):
        return 'NotImplemented'


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class VRAE(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    """
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0.3, optimizer='Adam', loss='MSELoss',
                 cuda=False, print_every=100, clip=True, max_grad_norm=5, dload='.'):

        super(VRAE, self).__init__()


        self.dtype = torch.FloatTensor
        self.use_cuda = cuda

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False


        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor


        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload

        if self.use_cuda:
            self.cuda()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)

        return x_decoded, latent

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss

    def compute_loss(self, X):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration

        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, _ = self(x)
        loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn)

        return loss, recon_loss, kl_loss, x


    def _train(self, train_loader):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times

        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()

        epoch_loss = 0
        t = 0

        for t, X in enumerate(train_loader):

            # Index first element of array to return tensor
            X = X[0]

            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            X = X.permute(1,0,2)

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, _ = self.compute_loss(X)
            loss.backward()

            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)

            # accumulator
            epoch_loss += loss.item()

            self.optimizer.step()

            if (t + 1) % self.print_every == 0:
                print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, loss.item(),
                                                                                    recon_loss.item(), kl_loss.item()))

        print('Average loss: {:.4f}'.format(epoch_loss / t))


    def fit(self, dataset, save = False):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`

        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """

        train_loader = DataLoader(dataset = dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)

        for i in range(self.n_epochs):
            print('Epoch: %s' % i)

            self._train(train_loader)

        self.is_fitted = True
        if save:
            #self.save('model.pth')
            self.save('model_params.pth')


    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function

        :param x: input batch tensor
        :return: intermediate latent vector
        """
        return self.lmbd(
                    self.encoder(
                        Variable(x.type(self.dtype), requires_grad = False)
                    )
        ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function

        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, _ = self(x)

        return x_decoded.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')


    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                for t, x in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above

        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later

        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned

        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Parameters
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("task", help="name of task to perfom: 'train', 'test'")
    arg_parser.add_argument("test_path", help="a path to test dataset")
    arg_parser.add_argument("train_path", help="a path to train dataset")
    args = arg_parser.parse_args()

    # Parse arguments
    task = args.task
    test_path = args.test_path
    train_path = args.train_path

    # Assign device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #dload = './model_dir'  # download directory
    dload = '/home/maintenance/IMUpaper'

    # configuration
    hidden_size = 90
    hidden_layer_depth = 3
    latent_length = 20
    batch_size = 32
    learning_rate = 0.0005
    n_epochs = 90
    dropout_rate = 0.3
    optimizer = 'Adam'  # options: Adam, SGD
    cuda = True  # options: True, False
    print_every = 30
    clip = True  # options: True, False
    max_grad_norm = 5
    loss = 'SmoothL1Loss'  # options: SmoothL1Loss, MSELoss
    block = 'LSTM'  # options: LSTM, GRU

    #MODE_LABELS = ['handheld', 'swing', 'texting', 'unknown']
    #MODE_LABELS = ['pocket', 'swing', 'texting', 'talking', 'Unknown']
    MODE_LABELS = ['pocket', 'swing', 'texting', 'talking']

    # Load train and test files - including handheld and scanning
    #X_train, y_train = load_into_x_y(train_path)
    #X_val, y_val = load_into_x_y(test_path)

    # Load train and test files - old data
    X_train, y_train = load_into_x_y_old_data(train_path)
    X_val, y_val = load_into_x_y_old_data(test_path)

    # X_train, X_val, y_train, y_val = open_data('data', ratio_train=0.9)

    # n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]  # using one-hot
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], len(np.unique(y_train))

    # transformation of data into torch tensors
    trainXT = torch.from_numpy(X_train).float()
    # trainXT = trainXT.transpose(1, 2).float()  # input is (N, Cin, Lin) = Ntimesteps, Nfeatures, 128
    trainyT = torch.from_numpy(y_train).float()
    valXT = torch.from_numpy(X_val).float()
    # valXT = valXT.transpose(1, 2).float()
    valyT = torch.from_numpy(y_val).float()

    train_dataset = TensorDataset(trainXT)
    test_dataset = TensorDataset(valXT)

    num_classes = len(np.unique(y_train))
    base = np.min(y_train)  # Check if data is 0-based
    if base != 0:
        y_train -= base
    y_val -= base


    sequence_length = X_train.shape[1]
    number_of_features = X_train.shape[2]

    vrae = VRAE(sequence_length=sequence_length,
                number_of_features=number_of_features,
                hidden_size=hidden_size,
                hidden_layer_depth=hidden_layer_depth,
                latent_length=latent_length,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                dropout_rate=dropout_rate,
                optimizer=optimizer,
                cuda=cuda,
                print_every=print_every,
                clip=clip,
                max_grad_norm=max_grad_norm,
                loss=loss,
                block=block,
                dload=dload)

    if task == 'test':
        # handheld and scanning
        #vrae.load(
        #    '/home/maintenance/projects/Unseen Mode 2/timeseries-clustering-vae-master/timeseries-clustering-vae-master/vrae_o/model_dir/vrae.pth')

        # 4 known labels
        vrae.load('/home/maintenance/IMUpaper/vrae_4classes.pth')

        num_params = vrae.get_num_params()

        z_run = vrae.transform(test_dataset)
        t_run = vrae.transform(train_dataset)

        # let's try to reduce to 2/3 samples for speedup decision
        sr = 1
        y_val_s = y_val[0::sr]
        y_train_s = y_train[0::sr]
        z_run_s = z_run[0::sr]
        t_run_s = t_run[0::sr]
        y_val_int_s = np.concatenate(y_val_s)
        y_train_int_s = np.concatenate(y_train_s)

        from pyod.models.knn import KNN

        # K=5
        tic()
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(t_run_s)
        z_test_pred = clf.predict(z_run_s)
        z_test_scores = clf.decision_function(z_run_s)
        correctly_detected_unknowns = np.sum(z_test_pred == 1)
        print('unsupervised:', correctly_detected_unknowns, correctly_detected_unknowns / len(z_test_pred))
        toc()

        # t-sne: let's try to reduce to 2/3 dimensions, with scaling
        # Train
        plot_tsne(t_run_s, y_train_int_s, dim=3, perplexity=30.0, scale_data=False)
        plot_tsne(t_run_s, y_train_int_s, dim=2, perplexity=30.0, scale_data=False)
        # Test
        tz_run_s = np.concatenate((t_run_s, z_run_s), axis=0)
        y_train_test_int_s = np.concatenate((y_train_int_s, y_val_int_s), axis=0)
        plot_tsne_plus_val(tz_run_s, y_train_test_int_s, dim=2, perplexity=30.0, scale_data=False)

        # PCA
        plot_pca_moons(t_run_s, y_train_int_s)
        plot_pca_moons(z_run_s, y_val_int_s)

        # Clustering
        plot_clustering(z_run_s, y_val_int_s, engine='matplotlib', download=False)
        # plot_clustering(z_run, y_val, engine='plotly', download = False)
        print('END test task')

    if task == 'train':

        # If the model has to be saved, with the learnt parameters use:
        vrae.fit(train_dataset, save=True)

        # If the latent vectors have to be saved, pass the parameter `save`
        z_run = vrae.transform(test_dataset, save=True)

        # To save a model, execute:
        # Handheld and scanning
        #vrae.save('vrae_handeling_scanning_model.pth')

        vrae.save('vrae_4classes.pth')

        # To load a presaved model, execute:
        # vrae.load('vrae.pth')

        # If plotly to be used as rendering engine, uncomment below line
        # plot_clustering(z_run, y_val, engine='plotly', download = False)
        plot_clustering(z_run, y_train, engine='matplotlib', download=True)
        #print('END train task')


