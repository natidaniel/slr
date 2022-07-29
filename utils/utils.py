import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report

# configurations
MODE_LABELS = ['pocket','swing','texting','talking','Unknown']
#MODE_LABELS = ['pocket','swing','texting','talking']
mode = ['devicemode']
SAMPLE_FREQ = 50
FILE_MARGINES = 2*SAMPLE_FREQ


def SignleLoader(root, file):
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
    data['SmartphoneMode'] = MODE_LABELS[-1]  ## 'cnn

    #data['devicemode'] = len(MODE_LABELS)
    data['devicemode'] = len(MODE_LABELS) - 1

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


def FilesLoader(filesLoc):
    print('Loading files from: ', filesLoc)
    return pd.concat([SignleLoader(filesLoc, f) for f in os.listdir(filesLoc) if f.lower().endswith('.csv')])


def indices_to_one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def toLstmFormat(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:, 1:7])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5)
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y


def toLstmFormatAcc(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:, 1:4])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5) #5
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y


def toLstmFormatWithFeatures(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:, [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 20, 21]])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5)
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y


def toLstmFormatWithFeaturesAcc(data, timestep):
    assert 0 < timestep < data.shape[0]
    xdata = np.asarray(data.iloc[:, [1, 2, 3, 14, 15]])
    ydata = np.asarray(data[mode])
    ydata = indices_to_one_hot(ydata, 5)  # 4 is number of modes
    x = np.array([xdata[start:start + timestep] for start in range(0, xdata.shape[0] - timestep)])
    y = ydata[:len(x)]
    return x, y


def load_into_x_y(dirname, TimeStep=32):
    data = FilesLoader(dirname)
    x, y = toLstmFormatAcc(data, TimeStep)
    return x, y


def plot_umap(X_embed, y_plot):
    colors = ['r', 'g', 'b', 'y', 'm']
    plt.figure(figsize=(12, 12))
    for i, c in enumerate(colors):
        plt.plot(X_embed[y_plot == i, 0], X_embed[y_plot == i, 1], c + '.')
    plt.show()


def quick_plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()