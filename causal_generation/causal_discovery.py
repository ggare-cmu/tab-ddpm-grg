import numpy as np

import os 
import sys

import subprocess

import utils

# sys.path.append(os.path.abspath('../ENCO-grg'))
# from ENCO-grg.causal_graphs.graph_definition import CausalDAGDataset


import networkx as nx
import pandas as pd


import zero



def drawAdjGraph(acyclic_adj_matrix, variables, classes, graph_path):

    G = pd.DataFrame(acyclic_adj_matrix, index = variables+classes, columns = variables+classes)
    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)

    drawGraph(G, graph_path)


def drawGraph(Graph, graph_path = ".", ):


    labels = nx.get_edge_attributes(Graph, "weight")
    
    #Change float precision
    for k,v in labels.items():
        labels[k] = f'{v:0.2f}'

    A = nx.nx_agraph.to_agraph(Graph)        # convert to a graphviz graph
    A.layout(prog='dot')            # neato layout
    #A.draw('test3.pdf')

    root_nodes = np.unique([e1 for (e1, e2), v in labels.items()])
    root_nodes_colors = {}

    for idx, node in enumerate(root_nodes):
        color =  "#"+''.join([hex(np.random.randint(0,16))[-1] for i in range(6)])
        root_nodes_colors[node] = color

    for (e1, e2), v in labels.items():
        edge = A.get_edge(e1,e2)
        edge.attr['weight'] = v
        edge.attr['label'] = str(v)
        # edge.attr['color'] = "red:blue"
        edge.attr['color'] = root_nodes_colors[e1]
        
    A.draw(graph_path,
            args='-Gnodesep=1.0 -Granksep=9.0 -Gfont_size=1', prog='dot' )  


def get_combined_features_labelOHE(dataset, classes, num_features):

    data_features = dataset.get_features('train')
    data_labels = dataset.y['train']

    assert np.array_equal(np.unique(data_labels), np.arange(len(classes))), "Error! Data labels not in correct format"

    #convert to one-hot encoding
    # data_labels_ohe = np.zeros((data_labels.shape[0], len(classes)))
    data_labels_ohe = np.zeros((data_labels.shape[0], len(classes)), dtype = data_features.dtype)
    data_labels_ohe[np.arange(data_labels.shape[0]), data_labels] = 1
    assert np.array_equal(data_labels_ohe.argmax(1), data_labels), "Error! One-hot-encoding incorrectly done."

    data_obs = np.concatenate((data_features, data_labels_ohe), axis = 1)
    assert data_obs.shape == (data_features.shape[0], num_features), "Error! Dataset not in correct format."

    return data_obs


def performCausalDiscovery(args, raw_config, dataset, variables, classes, feature_type):

    zero.improve_reproducibility(raw_config['seed'])

    results_dir = os.path.join(raw_config['parent_dir'], "causal_graph")
    utils.createDirIfDoesntExists(results_dir)

    all_feature_types = list(feature_type.values())
    if 'continous' in all_feature_types:
        data_type = np.float32
    elif np.array_equal(np.unique(all_feature_types), ['binary', 'binary-class']):
        data_type = np.uint8
    else:
        data_type = np.int32

    num_features = len(variables) + len(classes)

    adj_matrix = np.zeros((num_features, num_features), dtype=np.uint8)

    data_int = np.zeros((num_features, 1, num_features), dtype=data_type)

    data_obs = get_combined_features_labelOHE(dataset, classes, num_features)

    data_features = dataset.get_features('train')
    
    if data_obs.dtype != data_features.dtype or data_obs.dtype != data_type:
        print(f"Setting dtype of data from {data_features.dtype} to {data_type}")
        data_obs = data_obs.astype(data_type)
    print(f"data_obs.dtype {data_obs.dtype}")

    data_path = os.path.join(results_dir, "train_data.npz")

    np.savez(data_path, data_obs = data_obs, data_int = data_int, adj_matrix = adj_matrix, 
            vars_list = variables, class_list = classes, feature_type = feature_type)


    causal_discovery_args = [

                    "--seed", "42", #//"43", //"14", //"7", //"14", //"43", //"44", //"42",
                    "--apply_adj_matrix_mask", #//#GRG
                    
                    "--use_flow_model", #//help='If True, a Deep Sigmoidal Flow will be used as model if'
                    "--sample_size_obs", "500000", #//"5000",
                    #// "--num_epochs", "5", //"30",
                    
                    "--graph_files", os.path.abspath(data_path), #f"{data_path}", 
                    #// "--graph_files", "datasets/ModelBiomarkers-max_TrainUpsampledPatient.npz", 
                    "--max_inters", "1", #//"10", //"0", //"-1",
                    "--lambda_sparse", "0.05", #"0.1", #//"0.0004", //"0.004", //"0.05", //"0.1", //"0.0001", //"0.1", //"0.4", //"0.01", //"0.4", //"0.125", //"0.1", //"0.0001", //"0.0004", //"0.01", //"0.01", //"0.04", //"0.4", //"0.004",
                    #// "--lr_model", "1e-2", //"3e-2", //"1e-2", //"5e-3", //hear-disease-binary lr_model = "1e-2"
                    #// "--lr_gamma", "2e-6", //"5e-3", //"2e-2",
                    #// "--lr_theta", "1e-6", //"2e-2", //"1e-1",
                    
                    "--save_model",
                    "--batch_size", "32", #//"2", //"32", //"16", //"32", //"32", //"16", //"64", //"16", //"32", //"64", //"128",
                    "--checkpoint_dir", os.path.abspath(results_dir), #f"{results_dir}",
            ]

    run_args = ["python", "/home/grg/Research/ENCO_grg/run_exported_graphs_grg.py"] + causal_discovery_args

    # subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)
    subprocess.run(run_args, check=True)

    print(f"Finished causal discovery.")


    ##Read discovered graph

    acyclic_adj_matrix = np.load(os.path.join(results_dir, f"binary_acyclic_matrix_001_train_data.npy"))

    print(f"acyclic_adj_matrix.shape = {acyclic_adj_matrix.shape}")
    
    #Draw graph
    DRAW_GRAPH = True
    if DRAW_GRAPH:

        graph_path = os.path.join(results_dir, f"discovered_adjacency_matrix.png")
        drawAdjGraph(acyclic_adj_matrix, variables, classes, graph_path)

    return acyclic_adj_matrix


