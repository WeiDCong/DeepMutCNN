import datetime
import pandas as pd
import numpy as np

import optuna
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense

from utils import set_seeds
from utils import PCC_RMSE, PCC
from utils import get_metrics
from utils import sequence_encoding

def train_model(model, X_train, X_val, y_train, y_val, params):
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n++++++ training starts at {start_time} ++++++\n')

    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU for training.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    batchsize = params['batch_size']
    lr = params['lr']
    patience = params['patience']
    epoch = params['epoch']
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer, 
        loss=PCC_RMSE, 
        metrics=[PCC]
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        mode='min', 
        restore_best_weights=True
    )

    history = model.fit(
        X_train, 
        y_train, 
        epochs=epoch, 
        batch_size=batchsize,
        validation_data=(X_val, y_val), 
        callbacks=[early_stopping],
    )

    train_loss = model.evaluate(X_train, y_train) 
    val_loss = model.evaluate(X_val, y_val)

    print(f'Train Loss: {train_loss[0]:.3f} | PCC: {train_loss[1]:.3f}')
    print(f'Val Loss: {val_loss[0]:.3f} | PCC: {val_loss[1]:.3f}')

    model.save(f"./models/{params['save_model']}")

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n++++++ training completed at {end_time} ++++++\n')

    return model

def hyper_search(train_enc_var, train_score, params):

    def objective(trial):

        set_seeds(42)
        gpus = tf.config.list_physical_devices('GPU')
        print(gpus)
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Using GPU for training.")
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")
    
        num_conv_layers = trial.suggest_int('num_conv_layers', 3, 5)

        filters = []
        kernels = []
        for i in range(num_conv_layers):
            filter = trial.suggest_categorical(f'filter_{i}', [16, 32, 64, 96, 128])
            kernel = trial.suggest_categorical(f'kernel_{i}', [3, 5, 7])

            filters.append(filter)
            kernels.append(kernel)

        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.1)
        lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
        dense_channel = trial.suggest_categorical('dense_channel', [96, 128, 256])

        kf = KFold(n_splits=5, shuffle=True)
        pcc_scores = []

        for train_index, val_index in kf.split(train_enc_var):
            X_train_fold, X_val_fold = train_enc_var[train_index], train_enc_var[val_index]
            y_train_fold, y_val_fold = train_score[train_index], train_score[val_index]

            model = Sequential()
            model.add(Conv1D(
                filters=filters[0], 
                kernel_size=kernels[0], 
                activation='relu', 
                input_shape=(X_train_fold.shape[1], X_train_fold.shape[2])
            ))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))

            for i in range(1, num_conv_layers):
                model.add(Conv1D(
                    filters=filters[i], 
                    kernel_size=kernels[i], 
                    padding='same', 
                    activation='relu'
                ))
                model.add(BatchNormalization())
                model.add(MaxPooling1D(pool_size=2))

            model.add(Dropout(dropout_rate))
            model.add(Flatten())
            model.add(Dense(dense_channel, activation='relu'))
            model.add(Dense(1, activation='linear'))

            optimizer = optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=PCC_RMSE, metrics=[PCC])

            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=30,
                mode='min', 
                restore_best_weights=True
            )

            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=100, 
                batch_size=32,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[early_stopping],
                verbose=0
            )

            best_val_pcc = max(history.history['val_PCC'])
            pcc_scores.append(best_val_pcc)
        average_pcc = np.mean(pcc_scores)
        return average_pcc
    
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n++++++ hyperparameter optimization starts at {start_time} ++++++\n')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25) 

    print("best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}\n")
    params['hyperopt'] = study.best_params
    params['lr'] = study.best_params['lr']

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n++++++ hyperparameter optimization completed at {end_time} ++++++\n')
    
    return params

def evaluate_model(model, X_test, y_test):
    y_true = y_test.reshape(-1)
    y_pred = model.predict(X_test).flatten()
    rmse, rp, rs = get_metrics(y_true, y_pred)
    print(f'\nRMSE: {rmse:.3f}, Rp:{rp:.3f}, Rs:{rs:.3f}\n')

def save_preictions(model, params):
    raw_data = pd.read_csv(params.dms_file)
    mutants = raw_data['mutant'].to_list()
    scores = raw_data['score'].to_list()

    enc_var = sequence_encoding(mutants, params.seq, params.offset)
    preds = model.predict(enc_var).flatten()

    save_preds = pd.DataFrame({
        'mutant': mutants,
        'score': scores,
        'prediction': preds,
    })
    file_name = params.dms_file.split("/")[-1].replace('.csv', '_pred.csv')
    save_preds.to_csv(f'./results/{file_name}', index=False)

