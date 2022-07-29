import keras
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import torch
import warnings
import tensorflow.python.framework.dtypes
import os
warnings.filterwarnings('ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Model, load_model
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from utils import *


if __name__ == "__main__":
    # Parameters
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("task", help="name of task to perfom: 'train', 'test'")
    arg_parser.add_argument("test_path", help="a path to test dataset")
    arg_parser.add_argument("train_path", help="a path to train dataset")
    arg_parser.add_argument("model", help="name of model to perfom: 'cnn', 'lstm'")
    args = arg_parser.parse_args()

    # Parse arguments
    task = args.task
    deep_model = args.model
    test_path = args.test_path
    train_path = args.train_path

    # Assign device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configuration
    if deep_model == 'cnn':
        MODE_LABELS = ['pocket', 'swing', 'texting', 'talking', 'Unknown']
        mode = ['devicemode']
        SAMPLE_FREQ = 50
        num_time_periods = 32
        num_sensors = 3
        FILE_MARGINES = 2 * SAMPLE_FREQ
        input_shape = num_time_periods * num_sensors

    if deep_model == 'lstm':
        print('prepare data')


    if task == 'train':
        if deep_model == 'cnn':
            # Simple CNN
            model_m = Sequential()
            model_m.add(Reshape((num_time_periods, num_sensors), input_shape=(input_shape,)))
            model_m.add(Conv1D(100, 3, activation='relu', input_shape=(num_time_periods, num_sensors)))
            model_m.add(Conv1D(100, 3, activation='relu'))
            model_m.add(MaxPooling1D(3))
            model_m.add(Dropout(0.3))
            model_m.add(Conv1D(160, 3, activation='relu'))
            model_m.add(Conv1D(160, 3, activation='relu'))
            model_m.add(GlobalAveragePooling1D())
            model_m.add(Dropout(0.5))
            model_m.add(Dense(len(MODE_LABELS), activation='softmax'))
            print(model_m.summary())

            # Model ckpts
            callbacks_list = [
                keras.callbacks.ModelCheckpoint(
                    filepath='ckpts/cnn_model.{epoch:02d}-{val_loss:.2f}.h5',
                    monitor='val_loss', save_best_only=True),
                keras.callbacks.EarlyStopping(monitor='acc', patience=1)]

            model_m.compile(loss='categorical_crossentropy',
                            optimizer='adam', metrics=['accuracy'])

        elif deep_model == 'lstm':
            # Model
            lstm_model = Sequential()
            lstm_model.add(LSTM(16, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
            lstm_model.add(Dense(64, activation='relu'))
            lstm_model.add(Dense(64, activation='relu'))
            lstm_model.add(Dropout(0.1))
            lstm_model.add(Dense(64, activation='relu'))
            lstm_model.add(Dense(64, activation='relu'))
            lstm_model.add(Dense(y_lstm_train.shape[1], activation='softmax'))
            lstm_model.summary()

            lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            early_stopping_monitor = EarlyStopping(patience=3)
        else:
            print('doesnt support other modesl')


        if deep_model == 'cnn':
            # Training Hyper-parameters
            BATCH_SIZE = 32
            EPOCHS = 5

            # Load train files
            x_train, y_train = load_into_x_y(train_path)
            # reshape for Keras
            x_train = x_train.reshape(x_train.shape[0], input_shape)

            # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
            history = model_m.fit(x_train,
                                  y_train,
                                  batch_size=BATCH_SIZE,
                                  epochs=EPOCHS,
                                  callbacks=callbacks_list,
                                  validation_split=0.2,
                                  verbose=1)

            # Save model
            model_m.save("smartphone_model_CNN.hdf5")


            # Plot Training results
            plt.figure(figsize=(6, 4))
            plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
            plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
            plt.plot(history.history['loss'], 'r--', label='Loss of training data')
            plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
            plt.title('Model Accuracy and Loss')
            plt.ylabel('Accuracy and Loss')
            plt.xlabel('Training Epoch')
            plt.ylim(0)
            plt.legend()
            plt.show()

            # Print confusion matrix for training data
            y_pred_train = model_m.predict(x_train)
            # Take the class with the highest probability from the train predictions
            max_y_pred_train = np.argmax(y_pred_train, axis=1)
            max_y_pred_train_c = np_utils.to_categorical(max_y_pred_train, len(MODE_LABELS))
            # print classification report
            print(classification_report(y_train, max_y_pred_train_c))
            # print CM
            max_y_train = np.argmax(y_train, axis=1)
            matrix_train = metrics.confusion_matrix(max_y_train, max_y_pred_train)
            # Normalise
            cmn = matrix_train.astype('float') / matrix_train.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=MODE_LABELS, yticklabels=MODE_LABELS)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show(block=False)
            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix_train,
                        cmap='coolwarm',
                        linecolor='white',
                        linewidths=1,
                        xticklabels=MODE_LABELS,
                        yticklabels=MODE_LABELS,
                        annot=True,
                        fmt='d')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            print('stop')

        elif deep_model == 'lstm':
            history = lstm_model.fit(X_lstm_train, y_lstm_train, validation_split=0.2, epochs=10,
                                     callbacks=[early_stopping_monitor])
            quick_plot_history(history)
        else:
            print('doesnt support other modesl')

    if task == 'test':
        if deep_model == 'cnn':
            # Load test files
            x_test, y_test = load_into_x_y(test_path)
            # reshape for Keras
            x_test = x_test.reshape(x_test.shape[0], input_shape)

            # Load model
            model_m = load_model("smartphone_model_CNN.hdf5")
            model_m.summary()
            score = model_m.evaluate(x_test, y_test, verbose=1)
            print('\nAccuracy on test data: %0.2f' % score[1])
            print('\nLoss on test data: %0.2f' % score[0])

            # Plot confusion matrix
            y_pred_test = model_m.predict(x_test)
            # Take the class with the highest probability from the test predictions
            max_y_pred_test = np.argmax(y_pred_test, axis=1)
            max_y_test = np.argmax(y_test, axis=1)

            matrix_test = metrics.confusion_matrix(max_y_test, max_y_pred_test)
            # Normalise
            cmn = matrix_test.astype('float') / matrix_test.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=MODE_LABELS, yticklabels=MODE_LABELS)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show(block=False)
            plt.figure(figsize=(6, 4))
            sns.heatmap(matrix_test,
                        cmap='coolwarm',
                        linecolor='white',
                        linewidths=1,
                        xticklabels=MODE_LABELS,
                        yticklabels=MODE_LABELS,
                        annot=True,
                        fmt='d')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            print(classification_report(max_y_test, max_y_pred_test))

            # Random prediction
            test_record = x_test[1].reshape(1, input_shape)
            keras_prediction = np.argmax(model_m.predict(test_record), axis=1)
            print('\nPrediction:\t', keras_prediction[0])
            print('\nTruth:\t\t', MODE_LABELS[np.argmax(y_test[1])])
        elif deep_model == 'lstm':
            y = y_[5:-1]
            preds = lstm_model.predict(X_lstm_test)
            preds_cat = np.argmax(preds, axis=1)
            # building a map of result to activity
            result = np.unique(preds_cat).tolist()
            expected = np.unique(y).tolist()
            combined = list(zip(result, expected))
            conf_map = dict(combined)
            results = [x for x in preds_cat]
            print('model accuracy on test :', accuracy_score(y, results) * 100)