import sys
import os
import random
import pandas as pd
import numpy as np

from os.path import abspath
import warnings
warnings.filterwarnings("ignore")

module_path = abspath("nn4dms")
if module_path not in sys.path:
    sys.path.append(module_path)
import nn4dms.encode as enc

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

import tensorflow as tf

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf, 'set_random_seed'):
        tf.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def find_common_rows_index(df1, df2):
    rows1 = set(df1.apply(tuple))
    rows2 = set(df2.apply(tuple))
    common_rows = rows1.intersection(rows2)
    common_indexes = df1[df1.apply(tuple).isin(common_rows)].index.tolist()
    return common_indexes

def sequence_encoding(mutants, sequence, offset):
    encoding = enc.encode(
        encoding="one_hot,aa_index", 
        variants=mutants, 
        wt_aa=sequence, 
        wt_offset=offset
        )
    return encoding

def read_data(dms_file, sequence, offset):
    raw_data = pd.read_csv(dms_file)
    testset = raw_data.sample(frac=0.1, random_state=1, replace=False) 
    common_indices = raw_data[raw_data.isin(testset.to_dict(orient='list')).all(axis=1)].index
    trainset = raw_data.drop(common_indices)

    train_mutants = trainset['mutant'].to_list()
    test_mutants = testset['mutant'].to_list()
    train_score = np.array(trainset['score'], dtype=np.float32).reshape(-1,1)
    test_score = np.array(testset['score'], dtype=np.float32).reshape(-1,1)

    train_enc_var = sequence_encoding(train_mutants, sequence, offset)
    test_enc_var = sequence_encoding(test_mutants, sequence, offset)

    return train_enc_var, train_score, test_enc_var, test_score

def PCC(true, pred):
    fsp = pred - tf.keras.backend.mean(pred)
    fst = true - tf.keras.backend.mean(true)
    devP = tf.keras.backend.std(pred)
    devT = tf.keras.backend.std(true) 
    pcc = tf.keras.backend.mean(fsp * fst) / (devP * devT)
    return pcc

def RMSE(true, pred):
    mse = tf.keras.backend.mean(tf.keras.backend.square(pred - true), axis=-1)
    rmse = tf.keras.backend.sqrt(mse)
    return rmse

def PCC_RMSE(true, pred):
    alpha = 0.8
    rmse = RMSE(true, pred)
    pcc = 1.0 - PCC(true, pred)
    loss = alpha * pcc + (1 - alpha) * rmse   
    return loss

def get_metrics(true, pred):
    rp, _ = pearsonr(true, pred)
    rs, _ = spearmanr(true, pred)
    rmse  = np.sqrt(mean_squared_error(true, pred))
    return rmse, rp, rs