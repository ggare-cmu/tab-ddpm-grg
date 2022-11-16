from copy import deepcopy
import torch
import os
import numpy as np
import zero

#To address import errors 
import sys
sys.path.append("./")

from causal_gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from utils_train import get_model, make_subdataset, update_ema
import lib
import pandas as pd

from causal_discovery import get_combined_features_labelOHE

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            # out_dict[k] = out_dict[k].long().to(self.device)
            # out_dict[k] = out_dict[k].float().to(self.device)
            out_dict[k] = out_dict[k].to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1



def fitModel(dataset, T_dict, 
    model_params, model_type, device, 
    batch_size, gaussian_loss_type, num_timesteps, 
    scheduler, lr, weight_decay, steps,
    parent_dir, node_var):

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)



    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, f'{node_var}_loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, f'{node_var}_model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, f'{node_var}_model_ema.pt'))

    return diffusion



def train(
    parent_dir,
    dataset,
    graph_params,
    variables,
    classes,
    feature_type,
    features_list,
    causal_feature_mapping,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False
):
    # real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    # T = lib.Transformations(**T_dict)

    # dataset = make_dataset(
    #     real_data_path,
    #     T,
    #     num_classes=model_params['num_classes'],
    #     is_y_cond=model_params['is_y_cond'],
    #     change_val=change_val
    # )


    num_features = len(variables) + len(classes)

    G, sorted_variables, node_parents, root_nodes, unconnected_nodes = graph_params["G"], graph_params["sorted_variables"], graph_params["node_parents"], graph_params["root_nodes"], graph_params["unconnected_nodes"]

    
    train_data_obs = get_combined_features_labelOHE(dataset, classes, num_features)

    trained_models = {}

    for node_var in sorted_variables:

        if node_var in unconnected_nodes:
            print(f"Skipping unconnected node {node_var}")

            train_labels = train_data_obs[:, causal_feature_mapping[node_var]]
            train_label_classes = np.unique(train_labels)
            trained_models[node_var] = {'label_class': train_label_classes}

            continue
        elif node_var in root_nodes:
            print(f"Skipping root node {node_var}")

            train_labels = train_data_obs[:, causal_feature_mapping[node_var]]
            train_label_classes = np.unique(train_labels)
            trained_models[node_var] = {'label_class': train_label_classes}

            continue

        parents = node_parents[node_var]
        print(f"Train for node {node_var} with parents {parents}")


        parents_idx = [causal_feature_mapping[v] for v in parents]
        parent_var_type = [feature_type[v] for v in parents]

        parent_var_features = train_data_obs[:, parents_idx]

        # train_labels = train_data[:, category_classes]
        node_var_label = train_data_obs[:, causal_feature_mapping[node_var]]
        
        node_var_type = feature_type[node_var]

        # train_label_classes = np.unique(train_labels)

        # if ADD_NOISE:
        #     train_sub_features = train_sub_features + rng.normal(0, 0.01, train_sub_features.shape)

        # if ADD_NOISE_VAR:
        #     noise = rng.normal(0.5, 0.5, train_sub_features.shape[0])
        #     noise = (noise - noise.min())/(noise.max() - noise.min())
        #     train_sub_features = np.concatenate((train_sub_features, noise[:, np.newaxis]), axis = 1)

        #     assert train_sub_features.shape[1] == len(parents_idx) + 1, "Error! Noise Variable added incorrectly."

        var_trainset =  make_subdataset(
                            parents,
                            parent_var_features,
                            parent_var_type,
                            node_var,
                            node_var_label,
                            node_var_type,
                            # change_val=False,
                            # apply_transformation=False,
                        )

        var_model_params = model_params.copy()
        var_model_params['d_in'] = var_trainset.n_features
        var_model_params['num_classes'] = var_trainset.n_classes
        var_model_params['is_y_cond'] = True
        var_model_params['y_size'] = var_trainset.y['train'].shape[1]
        
        print(f"Running var {node_var} with model_params {var_model_params}")

        ##Train model
        diffusion = fitModel(   
                    var_trainset, T_dict, 
                    var_model_params, model_type, device, 
                    batch_size, gaussian_loss_type, num_timesteps, 
                    scheduler, lr, weight_decay, steps,
                    parent_dir, node_var
                )


        
        
        #Train model

        #Save trained model
        # dump(model, os.path.join(reports_path, f"Gen_ML_Model_var_{node_var}.joblib")) 
        torch.save(diffusion.state_dict(), os.path.join(parent_dir, f'{node_var}_diffusion_model.pt'))

        trained_models[node_var] = {'model': diffusion, 'label_class': train_label_classes}


    



#Numpy new way to sample from distributions: Ref: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
from numpy.random import default_rng
rng = default_rng()


#For saving & loading sklearn model - Ref: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations
from joblib import dump, load

def train_generative_models(sorted_variables, node_parents, root_nodes, unconnected_nodes, train_data, category_classes, 
            causal_feature_mapping, feature_type_dict, hidden_layer_sizes = (128, 64, 32), max_iter = 200, 
            ADD_NOISE = False, ADD_NOISE_VAR = False, reports_path = "."):

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

        # if ADD_NOISE:
        #     train_sub_features = train_sub_features + rng.normal(0, 0.01, train_sub_features.shape)

        # if ADD_NOISE_VAR:
        #     noise = rng.normal(0.5, 0.5, train_sub_features.shape[0])
        #     noise = (noise - noise.min())/(noise.max() - noise.min())
        #     train_sub_features = np.concatenate((train_sub_features, noise[:, np.newaxis]), axis = 1)

        #     assert train_sub_features.shape[1] == len(parents_idx) + 1, "Error! Noise Variable added incorrectly."

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

