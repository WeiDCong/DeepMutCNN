#!/usr/bin/env python3
import argparse
import datetime

from sklearn.model_selection import train_test_split

from utils import set_seeds
from utils import PCC, PCC_RMSE
from utils import read_data
from model import build_model
from train import train_model, hyper_search, evaluate_model, save_preictions


def get_params(args):
    model_name = args.dms_file.split("/")[-1].replace('csv', 'h5')
    params = {
        'batch_size': 32,
        'patience': 25,
        'epoch': 120,
        'save_model': model_name
    }
    if args.model == 'default':
        params['model'] = 'default'
        params['lr'] = 1e-4    
    else:
        params['model'] = 'hyperopt'
    
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",
                        required=True,                        
                        help="wildtype amino acid sequence",
                        type=str)
    parser.add_argument("--offset",
                        required=True,
                        help="start index",
                        type=int)
    parser.add_argument("--dms_file",
                        required=True,
                        help="path to your DMS data file",
                        type=str)
    parser.add_argument("--model",
                        choices=['default', 'hyperopt'],
                        default="default",
                        help="default: using default model setup; hyperopt: search for optimal model setup",
                        type=str)
    
    args = parser.parse_args()
    set_seeds(42)

    if args.model == 'default':
        train_enc_var, train_score, test_enc_var, test_score = read_data(
            args.dms_file, 
            args.seq, 
            args.offset
            )
        train_enc_var, val_enc_var, train_score, val_score= train_test_split(
            train_enc_var, 
            train_score, 
            test_size=0.1, 
            )
        params = get_params(args)
        params['input_shape'] = train_enc_var.shape[1:]
        model = build_model(params)
        model = train_model(model, train_enc_var, val_enc_var, train_score, val_score, params)
        evaluate_model(model, test_enc_var, test_score)
        save_preictions(model, args)
    elif args.model == 'hyperopt':
        train_enc_var, train_score, test_enc_var, test_score = read_data(
            args.dms_file, 
            args.seq, 
            args.offset
            )
        train_enc_var, val_enc_var, train_score, val_score= train_test_split(
            train_enc_var, 
            train_score, 
            test_size=0.1, 
            )
        params = get_params(args)
        params['input_shape'] = train_enc_var.shape[1:]
        params = hyper_search(train_enc_var, train_score, params)
        model = build_model(params)
        model = train_model(model, train_enc_var, val_enc_var, train_score, val_score, params)
        evaluate_model(model, test_enc_var, test_score)
        save_preictions(model, args)
    else:
        raise ValueError('Invalid model parameters!')
    
    
