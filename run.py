#!/usr/bin/env python3
import argparse
import datetime

from sklearn.model_selection import train_test_split

from utility import set_seeds
from utility import PCC, PCC_RMSE
from utility import read_data
from model import build_model
from train import train_model, hyper_search, evaluate_model, save_predictions, inference


def get_params(args):
    model_name = args.dms_file.split("/")[-1].replace('csv', 'h5')
    params = {
        'batch_size': 32,
        'patience': 35,
        'epoch': 140,
        'save_model': model_name
    }
    if args.mode == 'default':
        params['mode'] = 'default'
        params['lr'] = 1e-4    
    else:
        params['mode'] = 'hyperopt'
    
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
    parser.add_argument("--mode",
                        choices=['default', 'hyperopt', 'inference'],
                        default="default",
                        help="default: using default model setup; hyperopt: search for optimal model setup; inference: using existing model to make predictions",
                        type=str)
    parser.add_argument("--model_file",
                        default=None,
                        help="path to the saved CNN model",
                        type=str)
    
    args = parser.parse_args()
    set_seeds(42)

    if args.mode == 'default':
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
        preds = save_predictions(model, args)
    elif args.mode == 'hyperopt':
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
        preds = save_predictions(model, args)
    elif args.mode == 'inference':
        if args.model_file is not None:
            inference(args)
    else:
        raise ValueError('Invalid mode parameters!')
    
    
