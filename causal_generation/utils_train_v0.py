import numpy as np
import os
import lib
import causal_data
from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)



def make_dataset(
    data_path: str,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool,
    apply_transformation: bool = True
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = causal_data.CausalDataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    D.cal_var_type()

    if change_val:
        D = causal_data.change_val(D)
    
    if apply_transformation:
        D = causal_data.transform_dataset(D, T, None)

    return D




def make_subdataset(
    parents: list,
    parent_var_features: list,
    parent_var_type: list,
    node_var: str,
    node_var_label: list,
    node_var_type: list,
    change_val=False,
    apply_transformation=False,
    # data_path: str,
    # T: lib.Transformations,
    # num_classes: int,
    # is_y_cond: bool,
    # change_val: bool
):

    X_cat_var = []
    X_num_var = []

    X_cat_t = []
    X_num_t = []
    # y_t = []
    # y_t = [node_var_label]
    y_t = node_var_label

    for p_var, p_var_feat, p_type in zip(parents, parent_var_features.T, parent_var_type):

        if p_type == "continous":
            X_num_t.append(p_var_feat)
            X_num_var.append(p_var)
        elif p_type == "categorical":
            X_cat_t.append(p_var_feat)
            X_cat_var.append(p_var)
        elif p_type == "binary":
            X_cat_t.append(p_var_feat)
            X_cat_var.append(p_var)
        elif p_type == "binary-class":
            X_cat_t.append(p_var_feat)
            X_cat_var.append(p_var)
            raise Exception(F"Error! Binary-class var {p_var} as parent feature for node = {node_var}")
        else:
            raise Exception(F"Error! Unsupported feature type = {p_type} for parent node {p_var}")


    if node_var_type == "continous":
        '''
        - `is_y_cond` -- false for regression, true for classification
        - `d_in` -- input dimension (not necessary, since scripts calculate it automatically)
        - `num_calsses` -- zero for regression, a number of classes for classification
        TaskType(enum.Enum):
            BINCLASS = 'binclass'
            MULTICLASS = 'multiclass'
            REGRESSION = 'regression'
        '''
        is_y_cond = False 
        num_classes = 0
        task_type = lib.TaskType('regression')

    # elif node_var_type == "categorical":
    #     is_y_cond = True 
    #     num_classes = np.unique(node_var_label).shape[0]
    #     if num_classes > 2:
    #         task_type = lib.TaskType('multiclass')
    #     else:
    #         task_type = lib.TaskType('binclass')

    elif node_var_type == "binary":
        is_y_cond = True 
        num_classes = np.unique(node_var_label).shape[0]
        assert num_classes == 2, "Error! Binary var with more than 2 classes."
        task_type = lib.TaskType('binclass')
    elif node_var_type == "binary-class":
        is_y_cond = True 
        num_classes = np.unique(node_var_label).shape[0]
        assert num_classes == 2, "Error! Binary-class var with more than 2 classes."
        task_type = lib.TaskType('binclass')
    else:
        raise Exception(F"Error! Unsupported feature type = {node_var_type}")

    if len(X_num_t) > 0:
        X_num_t = np.array(X_num_t).T
        assert X_num_t.shape[0] == parent_var_features.shape[0], "Error! X_num num of observations incorrect."

    if len(X_cat_t) > 0:
        X_cat_t = np.array(X_cat_t).T
        assert X_cat_t.shape[0] == parent_var_features.shape[0], "Error! X_cat num of observations incorrect."

    assert len(X_num_var) + len(X_cat_var) == len(parents), "Error! Parents not correctly distributed between X_num and X_cat"
    

    # classification
    if num_classes > 0:
        X_cat = {} if len(X_cat_t) > 0 or not is_y_cond else None
        X_num = {} if len(X_num_t) > 0 else None
        y = {} 

        for split in ['train']:
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if len(X_cat_t) > 0  else None
        X_num = {} if len(X_num_t) > 0 or not is_y_cond else None
        y = {}

        for split in ['train']:
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t


    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=task_type,
        n_classes=num_classes
    )

    if change_val:
        D = lib.change_val(D)
    
    if apply_transformation:
        D = causal_data.transform_dataset(D, T, None)

    return D


    