from ctypes import util
import numpy as np

import matplotlib.pyplot as plt

import torch

import os

import pandas as pd
import networkx as nx

from graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations

import utils


import random



def set_seed(seed = 42):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# #Relabel nodes
# vars_mapping = {
#     "$X_{1}$": "a0", "$X_{2}$": "a1", "$X_{3}$": "a2", "$X_{4}$": "a3", "$X_{5}$": "a4",
#     "$X_{6}$": "b0", "$X_{7}$": "b1", "$X_{8}$": "b2", "$X_{9}$": "b3", "$X_{10}$": "b4",
#     "$X_{11}$": "bo0", "$X_{12}$": "bo1", "$X_{13}$": "bo2",
#     "$X_{14}$": "pt0", "$X_{15}$": "pt1", "$X_{16}$": "pt2", "$X_{17}$": "pt3",
#     "$X_{18}$": "pl0", "$X_{19}$": "pl1", "$X_{20}$": "pl2",
#     "$X_{21}$": "i0", "$X_{22}$": "i1", "$X_{23}$": "i2", "$X_{24}$": "i3", "$X_{25}$": "i4",
#     "$X_{26}$": "pb0", "$X_{27}$": "pb1", "$X_{28}$": "pb2", "$X_{29}$": "pb3", "$X_{30}$": "pb4",
#     "$X_{31}$": "c0", "$X_{32}$": "c1", "$X_{33}$": "c2", "$X_{34}$": "c3", "$X_{35}$": "c4",
#     "$X_{36}$": "e0", "$X_{37}$": "e1", "$X_{38}$": "e2",
#     # "$X_{39}$": "s0", "$X_{40}$": "s1", "$X_{41}$": "s2", "$X_{42}$": "s3" #Severity classes
#     "$X_{39}$": "d0", "$X_{40}$": "d1", "$X_{41}$": "d2", "$X_{42}$": "d3", "$X_{43}$": "d4", "$X_{44}$": "d5", "$X_{45}$": "d6" #Disease classes
# }


# #Relabel nodes
# causal_feature_mapping = {
#     "a0":0, "a1":1, "a2":2, "a3":3, "a4":4,
#     "b0":5, "b1":6, "b2":7, "b3":8, "b4":9,
#     "bo0":10, "bo1":11, "bo2":12,
#     "pt0":13, "pt1":14, "pt2":15, "pt3":16,
#     "pl0":17, "pl1":18, "pl2":19,
#     "i0":20, "i1":21, "i2":22, "i3":23, "i4":24,
#     "pb0":25, "pb1":26, "pb2":27, "pb3":28, "pb4":29,
#     "c0":30, "c1":31, "c2":32, "c3":33, "c4":34,
#     "e0":35, "e1":36, "e2":37,
#     # "s0":38, "s1":39, "s2":40, "s3":41 #Severity classes
#     "d0":38, "d1":39, "d2":40, "d3":41, "d4":42, "d5":43, "d6":44 #Disease classes
# }


def drawGraph(Graph, reports_path = ".", ):


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
        
    A.draw(os.path.join(reports_path, f"adjacency_matrix_temp.png"),
            args='-Gnodesep=1.0 -Granksep=9.0 -Gfont_size=1', prog='dot' )  


def getSamplingOrder(adj_matrix, variables, logger, results_dir, DRAW_GRAPH = False):


    G = pd.DataFrame(adj_matrix, index = variables, columns = variables)
    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)
    
    #Draw graph

    # DRAW_GRAPH = False
    if DRAW_GRAPH:
        # nx.draw_networkx(G)
        # plt.savefig('Graph_temp.png')
        # plt.show()
        drawGraph(G, results_dir)

    #Node degrees 
    node_degrees = {}
    for var in variables:
        node_degrees[var]= G.degree[var]

    #Node in-degrees 
    node_in_degrees = {}
    for var in variables:
        node_in_degrees[var]= G.in_degree[var]

    in_degree_sort_idx = np.argsort(list(node_in_degrees.values()))
    ascending_in_degree_nodes =  np.array(variables.copy())[in_degree_sort_idx]

    #Node out-degrees 
    node_out_degrees = {}
    for var in variables:
        node_out_degrees[var]= G.out_degree[var]

    out_degree_sort_idx = np.argsort(list(node_out_degrees.values()))
    ascending_out_degree_nodes =  np.array(variables.copy())[out_degree_sort_idx]

    #Node parents
    node_parents = {}
    for var in variables:
        node_parents[var]= list(G.predecessors(var))

    unconnected_nodes = [v for v,d in node_degrees.items() if d == 0]
    logger.log(f"Unconnected nodes = {unconnected_nodes}")

    root_nodes = [n for n,d in G.in_degree() if d == 0 and G.degree(n) != 0] 
    logger.log(f"Root nodes = {root_nodes}")

    # nx.ancestors(G, 'd0')

    sorted_variables, edges, adj_matrix, sorted_idxs = sort_graph_by_vars(variables, adj_matrix = adj_matrix)

    # sorted_variables = np.array(variables)[sorted_idxs]

    # paths_dict = {}
    # for node in G:
    #     if G.out_degree(node)==0: #it's a leaf
    #         paths_dict[node] = nx.shortest_path(G, root, node)

    for var in sorted_variables:
        logger.log(f"{var}: {node_parents[var]}")

    return G, sorted_variables, node_parents, root_nodes, unconnected_nodes




from sklearn import tree, svm, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

def fitLargeMLP(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, 
        label_classes, feature_type = '',
        n_trial = 3, hidden_layer_sizes = (128, 64, 32), max_iter = 200, verbose = True):

    best_clf = None
    best_acc = -np.inf
    for idx in range(n_trial):

        # clf = MLPClassifier()

        if feature_type == "continous":
            clf = MLPRegressor(
                # hidden_layer_sizes = (128, 64),
                # hidden_layer_sizes = (128, 64, 32),
                hidden_layer_sizes = hidden_layer_sizes,
                learning_rate = "adaptive", #constant
                max_iter = max_iter,
                verbose = verbose,
            )
        else:
            clf = MLPClassifier(
                # hidden_layer_sizes = (128, 64),
                # hidden_layer_sizes = (128, 64, 32),
                hidden_layer_sizes = hidden_layer_sizes,
                learning_rate = "adaptive", #constant
                max_iter = max_iter,
                verbose = verbose,
            )
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        # #Map predictions to proper class labels; Here some class values can be missing
        # ml_predictions = label_classes[ml_predictions]

        # if feature_type == "continous":
        #     accuracy = clf.score(test_label_ft, gt_test_scores)
        # else:
        #     accuracy = (ml_predictions == gt_test_scores).mean()
        accuracy = clf.score(test_label_ft, gt_test_scores)
        print(f'[Trial-{idx}] ML model (MLP Large) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf

    if feature_type == "continous":
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = ml_predictions
    else:
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = clf.predict_proba(test_label_ft)


    # #Map predictions to proper class labels; Here some class values can be missing
    # ml_predictions = label_classes[ml_predictions]

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (MLP Large) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions




def fitRandomForest(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = RandomForestClassifier(n_estimators = 100) #The number of trees in the forest (default 100).
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (RandomForest) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (RandomForest) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions



def fitDecisionTree(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (DecisionTree) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (DecisionTree) accuracy = {accuracy}')


    
    return clf, accuracy, ml_predictions, ml_prob_predictions



def fitSVM(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        # clf = svm.SVC()
        clf = svm.SVC(probability = True) #Enable probability predictions
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (SVM) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (SVM) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions



def predLargeMLP(clf, test_label_ft, label_classes, feature_type):
    
    if feature_type == "continous":
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = ml_predictions
    else:
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = clf.predict_proba(test_label_ft)

    # #Map predictions to proper class labels; Here some class values can be missing
    # ml_predictions = label_classes[ml_predictions]

    return clf, ml_predictions, ml_prob_predictions




#For saving & loading sklearn model - Ref: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
from joblib import dump, load

def train_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, category_classes, 
            causal_feature_mapping, feature_type_dict, hidden_layer_sizes = (128, 64, 32), max_iter = 200, 
            ADD_NOISE = False, reports_path = "."):

    trained_models = {}

    for var in sorted_variables:

        if var in unconnected_nodes:
            print(f"Skipping unconnected node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue
        elif var in root_nodes:
            print(f"Skipping root node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue

        parents = node_parents[var]
        print(f"Train for node {var} with parents {parents}")


        parents_idx = [causal_feature_mapping[v] for v in parents]

        train_sub_features = train_data[:, parents_idx]

        # train_labels = train_data[:, category_classes]
        train_labels = train_data[:, causal_feature_mapping[var]]

        train_label_classes = np.unique(train_labels)

        if ADD_NOISE:
            train_sub_features = train_sub_features + rng.normal(0, 0.01, train_sub_features.shape)

        ##Train model
        model, accuracy, ml_predictions, ml_prob_predictions = fitLargeMLP(
                                    train_label_ft =  train_sub_features, 
                                    test_label_ft = train_sub_features, 
                                    gt_train_scores = train_labels, gt_test_scores = train_labels, 
                                    label_classes = train_label_classes,
                                    feature_type = feature_type_dict[var],
                                    n_trial = 3,
                                    # hidden_layer_sizes = (128, 64, 32),
                                    hidden_layer_sizes = hidden_layer_sizes,
                                    max_iter = max_iter,
                                    verbose = False
                                )

        #Train model

        #Save trained model
        dump(model, os.path.join(reports_path, f"Gen_ML_Model_var_{var}.joblib")) 

        trained_models[var] = {'model': model, 'label_class': train_label_classes}

    return trained_models



def load_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, 
            causal_feature_mapping, reports_path = "."):

    trained_models = {}

    for var in sorted_variables:

        if var in unconnected_nodes:
            print(f"Skipping unconnected node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue
        elif var in root_nodes:
            print(f"Skipping root node {var}")

            train_labels = train_data[:, causal_feature_mapping[var]]
            train_label_classes = np.unique(train_labels)
            trained_models[var] = {'label_class': train_label_classes}

            continue

        parents = node_parents[var]
        print(f"Load model for node {var} with parents {parents}")


        #Save trained model
        model = load(os.path.join(reports_path, f"Gen_ML_Model_var_{var}.joblib")) 

        trained_models[var] = {'model': model, 'label_class': train_label_classes}

    return trained_models



#Numpy new way to sample from distributions: Ref: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
from numpy.random import default_rng
rng = default_rng()

def generateNewData(trained_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
            category_classes, feature_type_dict, num_samples = 1000):

    generated_data_dict = {} 

    for var in sorted_variables:

        feature_type = feature_type_dict[var]

        if var in unconnected_nodes:
            print(f"Generating unconnected node {var}")

            # if feature_type == "binary":
            #     sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)

            # elif feature_type == "categorical":

            #     model_dict = trained_models[var]
            #     label_class = model_dict['label_class']

            #     sampled_var = rng.choice(label_class, size = num_samples)

            # elif feature_type == "continous":
                
            #     sampled_var = rng.uniform(low = 0, high = 1, size = num_samples)

            # else:
            #     raise Exception(F"Error! Unsupported feature type = {feature_type}")

            sampled_var = np.zeros(num_samples)
            print(f"Sampling constant zero for unconnected node {var}")

            sampled_var = sampled_var[:, np.newaxis]
            generated_data_dict[var] = sampled_var

            continue

        elif var in root_nodes:
            print(f"Generating root node {var}")

            if feature_type == "binary":
                sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)
            
            elif feature_type == "binary-class":
                # sampled_var = rng.binomial(n = 1, p = 0.5, size = num_samples)
                sampled_var = rng.uniform(low = 0, high = 1, size = num_samples)
                print(f"Error! Class var {var} in root node.")
            
            elif feature_type == "categorical":

                model_dict = trained_models[var]
                label_class = model_dict['label_class']

                sampled_var = rng.choice(label_class, size = num_samples)

            elif feature_type == "continous":
                
                sampled_var = rng.uniform(low = 0, high = 1, size = num_samples)

            else:
                raise Exception(F"Error! Unsupported feature type = {feature_type}")

            sampled_var = sampled_var[:, np.newaxis]
            generated_data_dict[var] = sampled_var
        
            continue

        parents = node_parents[var]
        print(f"Generating data for node {var} with parents {parents}")

        
        # test_features = np.array([generated_data[p] for p in parents]).T
        test_features = np.hstack([generated_data_dict[p] for p in parents])
        print(f'test_featurs.shape = {test_features.shape}')

        ##Eval model

        model_dict = trained_models[var]

        model = model_dict['model']
        label_class = model_dict['label_class']

        model, ml_predictions, ml_prob_predictions = predLargeMLP(model, test_features, label_class, feature_type)

        
        if feature_type == "binary":

            sampled_var = ml_predictions

        elif feature_type == "binary-class":
            
            # if len(category_classes) == 2:
            #     sampled_var = ml_predictions
            # else:
            #     assert ml_prob_predictions.shape[1] == 2, "Error! Prob pred dim should be 2 (binary)."
            #     sampled_var = ml_prob_predictions[:,1]

            assert ml_prob_predictions.shape[1] == 2, "Error! Prob pred dim should be 2 (binary)."
            sampled_var = ml_prob_predictions[:,1]

        elif feature_type == "categorical":

            sampled_var = ml_predictions

        elif feature_type == "continous":

            sampled_var = ml_prob_predictions

            # sampled_var = sampled_var.argmax(1)

        else:
            raise Exception(F"Error! Unsupported feature type = {feature_type}")


        sampled_var = sampled_var[:, np.newaxis]

        generated_data_dict[var] = sampled_var

    
    generated_data = np.hstack([generated_data_dict[v] for v in variables])

    return generated_data





from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix
from scipy.special import softmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def calScores(preds, prob_preds, targets, class_names, task, logger, binary_cross_entropy = False, skip_auc = False):

    labels = np.arange(len(class_names))
    

    accuracy = accuracy_score(targets, preds)

    if binary_cross_entropy:
        confusionMatrix = multilabel_confusion_matrix(targets, preds, labels = labels)
    else:
        confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    # confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    if binary_cross_entropy or skip_auc:
        auc = "-"
    else:
        auc = roc_auc_score(targets, prob_preds, average = "weighted", multi_class = "ovo") # multi_class = "ovr"
    precision = precision_score(targets, preds, average='weighted') #score-All average
    recall = recall_score(targets, preds, average='weighted') #score-All average
    f1 = f1_score(targets, preds, average='weighted') #score-All average
        
    classificationReport = classification_report(targets, preds, labels = labels, target_names = class_names, digits=5)

    logger.log(f"auc = {auc}")
    logger.log(f"accuracy = {accuracy}")
    logger.log(f"precision = {precision}")
    logger.log(f"recall = {recall}")
    logger.log(f"f1 = {f1}")
    logger.log(f"confusionMatrix = \n {confusionMatrix}")
    logger.log(f"classificationReport = \n {classificationReport}")


    results_dict = {}
    results_dict["auc"] = auc
    results_dict["accuracy"] = accuracy
    results_dict["precision"] = precision
    results_dict["recall"] = recall
    results_dict["f1"] = f1
    results_dict["confusionMatrix"] = confusionMatrix.tolist()
    results_dict["classificationReport"] = classificationReport

    return results_dict


def upsampleFeatures(labels, features):

    classes, count = np.unique(labels, return_counts = True)
    print(f"[Pre-Upsampling] classes, count = {classes, count}")   
    
    max_count = max(count)

    label_indices = []
    for c in classes:

        c_idx = np.where(labels == c)[0]
        assert np.unique(labels[c_idx]) == c, "Error! Wrong class index filtered."

        #Bug-GRG : Since we sample randomly some of the videos are never sampled/included. 
        # So, make sure to only sample additional required videos after including all videos at least once!
        #For the max count class, set replace to False as setting it True might exclude some samples from training
        # upsample_c_idx = np.random.choice(c_idx, size = max_count, replace = len(c_idx) < max_count)
        if len(c_idx) < max_count:
            # upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = len(c_idx) < max_count).tolist())
            upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = max_count > 2*len(c_idx)).tolist())
        else:
            upsample_c_idx = c_idx
        
        np.random.shuffle(upsample_c_idx)
        
        assert c_idx.shape == np.unique(upsample_c_idx).shape, "Error! Some videos where excluded on updampling."

        label_indices.extend(upsample_c_idx)

    assert len(label_indices) == max_count * len(classes)

    upsample_label_indices = label_indices

    # upsampled_features = features[label_indices, :]
    upsampled_features = features[label_indices]

    upsampled_labels = labels[label_indices]

    classes, count = np.unique(upsampled_labels, return_counts = True)
    print(f"[Post-Upsampling] classes, count = {classes, count}")   

    assert np.array_equal(count, max_count * np.ones(len(classes))), "Error! Upsampling didn't result in class-balance"

    return upsampled_labels, upsampled_features, upsample_label_indices



def main():

    exp_name = "generateContinousSamples"
    
    # task = 'heart-disease' #'heart-disease-binary' #'heart-disease' #'parity5' #'labor'
    # task = 'cifar-t1'
    task = 'higgs_small-t1'

    # causal_discovery_exp_dir = f"/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_{task}_TrainUpsampledPatient"
    # causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_4_Acyclic_{task}_TrainUpsampledPatient_T1"
    causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_{task}_TrainUpsampledPatient_T1"


    # set_seed()
    ShouldTrain = False #True
    load_model_dir = "/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_higgs-small-t1_TrainUpsampledPatient_T1/generateContinousSamples_10000"

    num_samples = 10000 #60000 #10000 #50 #200 #3000 #500 #1000
    # Binary_Features = False #True
    # features_type = 'continous' #categorical, binary, continous

    ADD_NOISE = False #False

    DRAW_GRAPH = False #False
    
    causal_mlp_hidden_layer_sizes = (128, 64, 32) #(32, 16) #(128, 64, 32)
    max_iter = 200 #200

    results_dir = os.path.join(causal_discovery_exp_dir, f"{exp_name}_{num_samples}")
    utils.createDirIfDoesntExists(results_dir)

    acyclic_adj_matrix_path = f"binary_acyclic_matrix_001_{task}_TrainUpsampledPatient.npy"

    acyclic_adj_matrix = np.load(os.path.join(causal_discovery_exp_dir, acyclic_adj_matrix_path))

    print(f"acyclic_adj_matrix.shape = {acyclic_adj_matrix.shape}")

    adj_matrix = acyclic_adj_matrix 
        
    ### Load training data
    
    train_data_path = f"datasets/{task}_TrainUpsampledPatient.npz"

    train_dataset = np.load(train_data_path, allow_pickle = True)

    train_data = train_dataset['data_obs']


    vars_list = train_dataset['vars_list']
    class_list = train_dataset['class_list']

    feature_type_dict = train_dataset['feature_type'].item()

    features_list = vars_list.tolist() + class_list.tolist()
    


    node_names_mapping = {}
    for idx, var in enumerate(features_list):
        # node_names_mapping[f"$X_\\{{idx}\\}$"] = var
        node_names_mapping["$X_{" + str(idx+1) + "}$"] = var

    causal_feature_mapping = {}
    for idx, var in enumerate(features_list):
        causal_feature_mapping[var] = idx

    num_vars = len(vars_list)
    num_classes = len(class_list)

    binary_cross_entropy = num_classes == 2

    num_categs = 1 + train_data.max(axis=0) #Bug-Fix GRG: Max should be taken along num_samples axis not num_vars axis
    assert len(num_categs) == adj_matrix.shape[0], "Error! Num categories does not match num of variables."
    new_categs_func = lambda i : num_categs[i]


    # class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others', ]
    class_names = class_list


    # category_classes = list(range(38, 38+num_classes))
    category_classes = list(range(num_vars, num_vars+num_classes))



    #Calculate Scores 
    reports_path = "."
    exp_name = "Using Synthetic Data"
    report_path = os.path.join(results_dir, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")



    ##Process adjacency matrix   

    REVERSE_OUTGOING_CLASS_EDGES = True

    if not REVERSE_OUTGOING_CLASS_EDGES:
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes
    else:
        #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
        # for s_idx in [38, 39, 40, 41]: #Severity classes
        # for s_idx in [38, 39, 40, 41, 42, 43, 44]: #Disease classes
        for s_idx in range(num_vars, num_vars+num_classes): #Disease classes
            outgoing_edges = adj_matrix[s_idx, :]
            outgoing_nodes = np.where(outgoing_edges == 1)
            adj_matrix[outgoing_nodes, s_idx] = 1
        
        #TODO-GRG: Check this
        #Now remove outgoing edges - otherwise this introduces cycles
        
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes

    # # TODO-GRG: Need to check if we need to transpose the adj_matrix or not!!!
    # # Transpose for mask because adj[i,j] means that i->j
    # mask_adj_matrices = adj_matrix.transpose(1, 2)

    # variables = list(range(45))
    # variables = list(vars_mapping.values())
    variables = features_list

    #Get sampling order
    G, sorted_variables, node_parents, root_nodes, unconnected_nodes = getSamplingOrder(adj_matrix, variables, logger, results_dir, DRAW_GRAPH)


    ### Train Generative Models

    #TODO-GRG: Add noise while learning generator model
    
    if ShouldTrain:
        trained_gen_models = train_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, 
                            category_classes, causal_feature_mapping, feature_type_dict, 
                            hidden_layer_sizes = causal_mlp_hidden_layer_sizes, 
                            max_iter = max_iter,
                            ADD_NOISE = ADD_NOISE,
                            reports_path = results_dir)
    else:
        trained_gen_models = load_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, 
                causal_feature_mapping, reports_path = load_model_dir) #results_dir)
        
    # num_samples = 500 #200 #3000 #500 #1000
    # Binary_Features = False #True
    # synthetic_data = generateNewData(trained_gen_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
    #         category_classes, feature_type_dict, num_samples)

    # # large_sample_size = 100*num_samples
    # # synthetic_data = generateNewData(trained_gen_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
    # #         category_classes, feature_type_dict, large_sample_size)

    # # syn_classes, syn_class_count = np.unique(synthetic_data[:, category_classes].argmax(1), return_counts = True)

    # #Save the generated synthetic data
    # np.save(os.path.join(results_dir, f"synthetic_data_{num_samples}.npy"), synthetic_data)

    ## Test the synthetic data

    print(f"train_data.shape = {train_data.shape}")

    train_features = train_data[:, :num_vars]

    train_labels = train_data[:, category_classes]

    UpSampleData = True #False
    if UpSampleData:
        upsam_train_labels, upsam_train_features, upsample_label_indices = upsampleFeatures(labels = train_labels.argmax(1), features = train_features) 
        train_labels = train_labels[upsample_label_indices]
        train_features = upsam_train_features

        assert train_labels.shape[0] == train_features.shape[0], "Error! Upsampled labels and feature count does not match."

    print(f"train_labels.shape = {train_labels.shape}")
    print(f"train_features.shape = {train_features.shape}")

    test_data_path = f"datasets/{task}_TestUpsampledPatient.npz"

    test_data = np.load(test_data_path)['data_obs']


    val_data_path = f"datasets/{task}_ValUpsampledPatient.npz"

    val_data = np.load(val_data_path)['data_obs']


    test_features = test_data[:, :num_vars]

    test_labels = test_data[:, category_classes]


    val_features = val_data[:, :num_vars]

    val_labels = val_data[:, category_classes]


    label_classes = np.unique(train_labels.argmax(1))



    ### Generate good synthetic data by evaluting Val set


    ##Train model
    org_val_model, org_val_accuracy, org_val_ml_predictions, org_val_ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  train_features, test_label_ft = val_features, 
                                # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = train_labels.argmax(1), gt_test_scores = val_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )

    logger.log(f"[MLPlarge] Val-Accuracy on original train set = {org_val_accuracy}")

    syn_val_accuracy = -1
    best_syn_val_accuracy = -1
    num_tries = 0
    while syn_val_accuracy <= org_val_accuracy:
        num_tries+=1
    # while syn_val_accuracy < org_val_accuracy:
        
        # synthetic_data = generateNewData(trained_gen_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
        #     category_classes, feature_type_dict, num_samples)

        # re_generate = True

        # while re_generate:
        large_sample_size = 100*num_samples
        synthetic_data = generateNewData(trained_gen_models, sorted_variables, variables, node_parents, root_nodes, unconnected_nodes, 
                category_classes, feature_type_dict, large_sample_size)

        syn_labels = synthetic_data[:, category_classes].argmax(1)
        syn_classes, syn_class_count = np.unique(syn_labels, return_counts = True)

        per_cls_sample = int(num_samples/len(syn_classes))
        c_idx_list = []
        for c in syn_classes:
            c_idx = np.where(syn_labels == c)[0]
            c_idx = np.random.choice(c_idx, per_cls_sample, replace = len(c_idx) < per_cls_sample)
            c_idx_list.extend(c_idx)

        if len(c_idx_list) < num_samples:
            c_idx_list.extend(c_idx[:num_samples - len(c_idx_list)])

        synthetic_data = synthetic_data[c_idx_list]
        assert synthetic_data.shape[0] == num_samples, "Error! Selecting samples from a large batch failed."

        syn_classes, syn_class_count = np.unique(synthetic_data[:, category_classes].argmax(1), return_counts = True)
        print(f"syn_classes, syn_class_count = {syn_classes, syn_class_count}")

        #Test synthetic data
        synthetic_train_data = np.concatenate((train_data, synthetic_data), axis = 0)
        print(f"synthetic_train_data.shape = {synthetic_train_data.shape}")
        
        synthetic_train_features = synthetic_train_data[:, :num_vars]

        synthetic_train_labels = synthetic_train_data[:, category_classes]


        syn_val_model, syn_val_accuracy, syn_val_ml_predictions, syn_val_ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  synthetic_train_features, test_label_ft = val_features, 
                                # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = val_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )
        
        logger.log(f"[MLPlarge] Val-Accuracy on synthetic + original train set = {syn_val_accuracy}")

        if syn_val_accuracy >= best_syn_val_accuracy:
            best_syn_val_accuracy = syn_val_accuracy
            best_synthetic_data = synthetic_data

        if num_tries > 5:
            break 
    
    synthetic_data = best_synthetic_data

    #Save the generated synthetic data
    np.save(os.path.join(results_dir, f"synthetic_data_{num_samples}.npy"), synthetic_data) 


    logger.log(f"only-synthetic class distn - {np.unique(synthetic_data[:, category_classes].argmax(1), return_counts = True)}")


    ### Evaluate the synthetic data on Test set









    ##Train model
    model, accuracy, ml_predictions, ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  train_features, test_label_ft = test_features, 
                                # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )

    logger.log(f"[MLPlarge] Accuracy on original train set = {accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    model_results_dict = calScores(preds = ml_predictions, prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger,
                binary_cross_entropy = binary_cross_entropy)
    

    # UpSampleSynthteicData = False #False

    runOnlyOnSynthetic = True
    if runOnlyOnSynthetic:
        only_synthetic_train_features = synthetic_data[:, :num_vars]

        only_synthetic_train_labels = synthetic_data[:, category_classes]

        logger.log(f"only-synthetic class distn - {np.unique(only_synthetic_train_labels.argmax(1), return_counts = True)}")

        if UpSampleData:
        # if UpSampleSynthteicData:
            upsam_synthetic_train_labels, upsam_synthetic_train_features, upsample_label_indices = upsampleFeatures(labels = only_synthetic_train_labels.argmax(1), features = only_synthetic_train_features) 
            only_synthetic_train_labels = only_synthetic_train_labels[upsample_label_indices]
            only_synthetic_train_features = upsam_synthetic_train_features

            assert only_synthetic_train_labels.shape[0] == only_synthetic_train_features.shape[0], "Error! Upsampled labels and feature count does not match."

        print(f"only_synthetic_train_labels.shape = {only_synthetic_train_labels.shape}")
        print(f"only_synthetic_train_features.shape = {only_synthetic_train_features.shape}")

        only_synthetic_model, only_synthetic_accuracy, only_synthetic_ml_predictions, only_synthetic_ml_prob_predictions = fitLargeMLP(
                                    train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    # features_type = "categorical",
                                    # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                    n_trial = 3, hidden_layer_sizes = (100),
                                    max_iter = max_iter,
                                    verbose = False
                                )

        logger.log(f"[MLPlarge] Accuracy on only synthetic set = {only_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        only_synthetic_model_results_dict = calScores(preds = only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-only_synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy,
                    skip_auc = True)


    synthetic_train_data = np.concatenate((train_data, synthetic_data), axis = 0)
    print(f"synthetic_train_data.shape = {synthetic_train_data.shape}")
    
    synthetic_train_features = synthetic_train_data[:, :num_vars]

    synthetic_train_labels = synthetic_train_data[:, category_classes]


    UpSampleSynthteicData = False #False
    # if UpSampleData:
    if UpSampleSynthteicData:
        upsam_synthetic_train_labels, upsam_synthetic_train_features, upsample_label_indices = upsampleFeatures(labels = synthetic_train_labels.argmax(1), features = synthetic_train_features) 
        synthetic_train_labels = synthetic_train_labels[upsample_label_indices]
        synthetic_train_features = upsam_synthetic_train_features

        assert synthetic_train_labels.shape[0] == synthetic_train_features.shape[0], "Error! Upsampled labels and feature count does not match."


    print(f"synthetic_train_labels.shape = {synthetic_train_labels.shape}")
    print(f"synthetic_train_features.shape = {synthetic_train_features.shape}")

    synthetic_model, synthetic_accuracy, synthetic_ml_predictions, synthetic_ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )

    logger.log(f"[MLPlarge] Accuracy on synthetic + original train set = {synthetic_accuracy}")

    # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
    synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger,
                binary_cross_entropy = binary_cross_entropy)


    ######### Random Forest ############

    runRandomForest = False
    if runRandomForest:

        ##Train model
        rf_model, rf_accuracy, rf_ml_predictions, rf_ml_prob_predictions = fitRandomForest(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[RandomForest] Accuracy on original train set = {rf_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        rf_model_results_dict = calScores(preds = rf_ml_predictions, prob_preds = torch.softmax(torch.Tensor(rf_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)

        rf_synthetic_model, rf_synthetic_accuracy, rf_synthetic_ml_predictions, rf_synthetic_ml_prob_predictions = fitRandomForest(
                                    train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[RandomForest] Accuracy on synthetic + original train set = {rf_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        rf_synthetic_model_results_dict = calScores(preds = rf_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(rf_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf"+"-synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)


        runOnlyOnSynthetic = True
        if runOnlyOnSynthetic:
            
            rf_only_synthetic_model, rf_only_synthetic_accuracy, rf_only_synthetic_ml_predictions, rf_only_synthetic_ml_prob_predictions = fitRandomForest(
                                        train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                        # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                        gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                        label_classes = label_classes,
                                        n_trial = 3
                                    )

            logger.log(f"[RandomForest] Accuracy on only synthetic set = {rf_only_synthetic_accuracy}")

            # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
            rf_only_synthetic_model_results_dict = calScores(preds = rf_only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(rf_only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                        targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf"+"-only_synthetic", logger = logger,
                        binary_cross_entropy = binary_cross_entropy,
                        skip_auc = True)




    ######### DecisionTree ############

    runDT = False
    if runDT:

        ##Train model
        dt_model, dt_accuracy, dt_ml_predictions, dt_ml_prob_predictions = fitDecisionTree(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[DT] Accuracy on original train set = {dt_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        dt_model_results_dict = calScores(preds = dt_ml_predictions, prob_preds = torch.softmax(torch.Tensor(dt_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)

        dt_synthetic_model, dt_synthetic_accuracy, dt_synthetic_ml_predictions, dt_synthetic_ml_prob_predictions = fitDecisionTree(
                                    train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[DT] Accuracy on synthetic + original train set = {dt_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        dt_synthetic_model_results_dict = calScores(preds = dt_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(dt_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)



        runOnlyOnSynthetic = True
        if runOnlyOnSynthetic:
            
            dt_only_synthetic_model, dt_only_synthetic_accuracy, dt_only_synthetic_ml_predictions, dt_only_synthetic_ml_prob_predictions = fitDecisionTree(
                                        train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                        # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                        gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                        label_classes = label_classes,
                                        n_trial = 3
                                    )

            logger.log(f"[DT] Accuracy on only synthetic set = {dt_only_synthetic_accuracy}")

            # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
            dt_only_synthetic_model_results_dict = calScores(preds = dt_only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(dt_only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                        targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-only_synthetic", logger = logger,
                        binary_cross_entropy = binary_cross_entropy,
                        skip_auc = True)

    ######### SVM ############

    runSVM = False
    if runSVM:

        ##Train model
        svm_model, svm_accuracy, svm_ml_predictions, svm_ml_prob_predictions = fitSVM(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[SVM] Accuracy on original train set = {svm_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        svm_model_results_dict = calScores(preds = svm_ml_predictions, prob_preds = torch.softmax(torch.Tensor(svm_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)

        svm_synthetic_model, svm_synthetic_accuracy, svm_synthetic_ml_predictions, svm_synthetic_ml_prob_predictions = fitSVM(
                                    train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[SVM] Accuracy on synthetic + original train set = {svm_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        svm_synthetic_model_results_dict = calScores(preds = svm_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(svm_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)



        runOnlyOnSynthetic = True
        if runOnlyOnSynthetic:
            
            svm_only_synthetic_model, svm_only_synthetic_accuracy, svm_only_synthetic_ml_predictions, svm_only_synthetic_ml_prob_predictions = fitSVM(
                                        train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                        # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                        gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                        label_classes = label_classes,
                                        n_trial = 3
                                    )

            logger.log(f"[SVM] Accuracy on only synthetic set = {svm_only_synthetic_accuracy}")

            # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
            svm_only_synthetic_model_results_dict = calScores(preds = svm_only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(svm_only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                        targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-only_synthetic", logger = logger,
                        binary_cross_entropy = binary_cross_entropy,
                        skip_auc = True)






    logger.log(f"Task = {task}")
    logger.log(f"only-synthetic class distn - {np.unique(synthetic_data[:, category_classes].argmax(1), return_counts = True)}")
    logger.log(f"[MLPlarge] Accuracy on original train set = {accuracy}")
    logger.log(f"[MLPlarge] Accuracy on only synthetic set = {only_synthetic_accuracy}")
    logger.log(f"[MLPlarge] Accuracy on synthetic + original train set = {synthetic_accuracy}")
    # logger.log(f"[RandomForest] Accuracy on original train set = {rf_accuracy}")
    # logger.log(f"[RandomForest] Accuracy on only synthetic set = {rf_only_synthetic_accuracy}")
    # logger.log(f"[RandomForest] Accuracy on synthetic + original train set = {rf_synthetic_accuracy}")
    # logger.log(f"[DT] Accuracy on original train set = {dt_accuracy}")
    # logger.log(f"[DT] Accuracy on only synthetic set = {dt_only_synthetic_accuracy}")
    # logger.log(f"[DT] Accuracy on synthetic + original train set = {dt_synthetic_accuracy}")
    # # logger.log(f"[SVM] Accuracy on original train set = {svm_accuracy}")
    # # logger.log(f"[SVM] Accuracy on only synthetic set = {svm_only_synthetic_accuracy}")
    # # logger.log(f"[SVM] Accuracy on synthetic + original train set = {svm_synthetic_accuracy}")

    # logger.log(f"{accuracy} \n{rf_accuracy} \n{dt_accuracy} \n{only_synthetic_accuracy} \n{rf_only_synthetic_accuracy} \n{dt_only_synthetic_accuracy} \n{synthetic_accuracy} \n{rf_synthetic_accuracy} \n{dt_synthetic_accuracy}")
    

    logger.close()

    pass


    return accuracy, only_synthetic_accuracy, synthetic_accuracy



if __name__ == "__main__":
    print("Started...")

    synthetic_accuracy = -1

    # while synthetic_accuracy < .82:
    # # while synthetic_accuracy < .99:
    # # while synthetic_accuracy < .40:
    #     accuracy, only_synthetic_accuracy, synthetic_accuracy = main()


    accuracy, only_synthetic_accuracy, synthetic_accuracy = main()
    
    
    print("Finished!")