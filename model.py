import datetime
import numpy as np
import optuna
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense

def build_model(params):
    input_shape = params['input_shape']
    model = Sequential()
    if params['mode'] == 'default':
        model.add(Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=64, kernel_size=7, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=7, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))
    elif params['mode'] == 'hyperopt':
        best_params = params['hyperopt']
        model.add(Conv1D(
            filters=best_params['filter_0'], 
            kernel_size=best_params['kernel_0'], 
            activation='relu', 
            input_shape=input_shape
            ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

        for i in range(1, best_params['num_conv_layers']):
            model.add(Conv1D(
                filters=best_params[f'filter_{i}'], 
                kernel_size=best_params[f'kernel_{i}'], 
                padding='same', 
                activation='relu'
            ))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))

        model.add(Dropout(best_params['dropout_rate']))
        model.add(Flatten())
        model.add(Dense(best_params['dense_channel'], activation='relu'))
        model.add(Dense(1, activation='linear'))

    return model
