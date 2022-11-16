import tomli
import shutil
import os
import argparse
from causal_train import train
from causal_sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import torch

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


def loadArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--discover', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)
    parser.add_argument('--apply_transformation', action='store_true',  default=True)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda:1')

    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    return args, raw_config, device



from causal_generation.causal_graph_lib import getSamplingOrder, preprocessAdjMatrix

import utils

import numpy as np

from utils_train import make_dataset


from causal_discovery import performCausalDiscovery, drawAdjGraph


def main():
    
    args, raw_config, device = loadArgs()

    zero.improve_reproducibility(raw_config['seed'])

    timer = zero.Timer()
    timer.run()

    model_params = raw_config['model_params']

    T_dict = raw_config['train']['T']
    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        data_path=raw_config['real_data_path'],
        T=T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=args.change_val,
        # apply_transformation=True,
        # apply_transformation=False
        apply_transformation=args.apply_transformation
    )

    variables = dataset.get_vars()
    classes = dataset.get_classes()
    feature_type = dataset.get_var_type()

    features_list = variables + classes

    causal_feature_mapping = {}
    for idx, node_var in enumerate(features_list):
        causal_feature_mapping[node_var] = idx


    if args.discover:
        performCausalDiscovery(
            args, 
            raw_config, 
            dataset, 
            variables, 
            classes, 
            feature_type
        )
    


    results_dir = raw_config['parent_dir']

    reports_path = os.path.join(results_dir, f"exp_report.log")
    logger = utils.Logger(reports_path)

    logger.log(f"Synthetic data generation report")


    causal_dir = os.path.join(results_dir, "causal_graph")
    adj_matrix = np.load(os.path.join(causal_dir, f"binary_acyclic_matrix_001_train_data.npy"))

    logger.log(f"adj_matrix.shape = {adj_matrix.shape}")

    adj_matrix = preprocessAdjMatrix(adj_matrix, num_vars = len(variables), num_classes = len(classes))


    #Draw graph
    DRAW_GRAPH = True
    if DRAW_GRAPH:

        graph_path = os.path.join(results_dir, f"processed_adjacency_matrix.png")
        drawAdjGraph(adj_matrix, variables, classes, graph_path)


    #Get sampling order
    graph_params = getSamplingOrder(adj_matrix, features_list, logger)

    

    if args.train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            dataset = dataset,
            graph_params = graph_params,
            variables = variables,
            classes = classes,
            feature_type = feature_type,
            features_list = features_list,
            causal_feature_mapping = causal_feature_mapping,
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val,
        )


    if args.sample:
        sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        elif raw_config['eval']['type']['eval_model'] == 'mlp':
            train_mlp(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                device=device
            )
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )

    print(f'Elapsed time: {str(timer)}')
    logger.log(f"Finished!")

    logger.close()



if __name__ == "__main__":
    print(f"Started...")
    main()
    print(f"Finished!")