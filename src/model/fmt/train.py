""" Agent models """
import os
import json
import argparse
import random
import math
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch
from torch import optim
from torch.utils.data import DataLoader
from src.model.fmt.model import FeatureManifoldTransformer, FMTConfig, DatasetDict
from collections import defaultdict

from src.tools.plot_embeddings import tsne_embeddings, umap_embeddings, EarlyStopping
from src.data.cins_cans_eda import TargetSet
from src.data.cins_cans_data_prep import load_data

# Live Data
# TRAIN_SIZE = 0.8
# EVAL_SIZE = 0.2
# DATASET = 'cins'
# QUESTION = 1
# D_QUESTIONS = ['cans_24']
D_QUESTIONS = ['cans_01', 'cans_06', 'cans_09', 'cans_10', 'cans_12', 'cans_18', 'cans_24', #'cans_17',
               'cins_01', 'cins_06', 'cins_09', 'cins_10', 'cins_11', 'cins_12', 'cins_14', 'cins_16']
NOISE_STD = 0.075
LAMBDA_REG = 0.06
DROPOUT = 0.0

# Universal
BASE_PROB = 0.4 # Only affects toy data
BATCH_SIZE = 128
RANDOM_SEED = 42
DEVICE = 'cpu'
MAX_EPOCHS = 500
PATIENCE = 500

def get_record(set: str, question: int, idx: int) -> dict:
    """
    Get a record from the dataset
    :param set: The dataset to get the record from
    :param idx: The index of the record
    :return: The record
    """
    _return = {}
    model_config = FMTConfig()
    model_config.target = f"{set}_{question:02d}"
    model_config.demos = {
        # 'age': 4,
        'gender': 2,
        'sbu_admit': 3,
        "sat_verbal_score": 4,
        "sat_math_score": 4,
        # 'english_first_language': 2,
    }
    model_config.knows = {
        'total_score': 21 if set == 'cins' else 25,
    }
    model_config.num_embd = 4
    model_config.num_heads = 4
    model_config.num_layers = 3
    _return['KNOW_RANKING'] = ""
    # * indicates the isolation group
    if idx == 0:
        _return['ISOLATION_GROUP'] = ['gender']
        # _return['KNOW_RANKING'] = 'total_score'
    elif idx == 1:
        _return['ISOLATION_GROUP'] = ['sbu_admit']
        # _return['KNOW_RANKING'] = 'total_score'
    elif idx == 2:
        _return['ISOLATION_GROUP'] = ['sat_math_score']
        # _return['KNOW_RANKING'] = 'total_score'
    elif idx == 3:
        _return['ISOLATION_GROUP'] = ['sat_verbal_score']
        # _return['KNOW_RANKING'] = 'total_score'
    else:
        raise ValueError("Invalid index.")
    _return['NAME'] = f"{model_config.target}-" + \
                      f"{_return['ISOLATION_GROUP'][0]}"
    if _return['KNOW_RANKING'] != "":
        _return['NAME'] += f"-{_return['KNOW_RANKING']}"
    return _return, model_config

def get_constants(type: str = "uniform", dataset: str = 'cins', question: int = 1):
    # Toy Data
    model_config = FMTConfig()
    model_config.demos = {
        'demo1': 3,
        'demo2': 4,
    }
    model_config.knows = {
        "know1": 4
    }
    model_config.target = 'target'

    _return = {}
    _return['ISOLATION_GROUP'] = ['demo1']
    _return['KNOW_RANKING'] = 'know1'
    if type == "uniform":
        _return['GROUP_SIZE'] = 20
        _return['SCALE'] = 0.0 # No scaling
        _return['UNI_DIF_SCALE'] = 0.0 # No scaling
        _return['NAME'] = "Uniform"
    elif type == "linear":
        _return['GROUP_SIZE'] = 40
        _return['SCALE'] = 0.05
        _return['UNI_DIF_SCALE'] = 0.0
        _return['NAME'] = "Linear"
    elif type == "uniform-dif":
        _return['GROUP_SIZE'] = 40
        _return['SCALE'] = 0.05
        _return['UNI_DIF_SCALE'] = 0.05
        _return['NAME'] = "Uniform-DIF"
    else: # Live data
        _return, model_config = get_record(dataset, question, 0)

    model_config.noise_std = NOISE_STD
    model_config.lambda_reg = LAMBDA_REG
    model_config.dropout = DROPOUT

    return _return, model_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_toy_data(opts: dict):
    train_set = []

    for i in range(3):
        for j in range(4):
            for l in range(4):
                # Adjust probability based on 'l', uniform dif on demo1
                prob = BASE_PROB + l * opts['SCALE'] + opts['UNI_DIF_SCALE'] * i

                # Create training samples
                num_ones = int(prob * opts['GROUP_SIZE'])
                num_zeros = opts['GROUP_SIZE'] - num_ones
                base_list = [1] * num_ones + [0] * num_zeros
                train_set.extend([(i, j, l, sample) for sample in base_list])

    return train_set, train_set

def get_labels(disp_data, consts):
    # Create individual label components for each isolation group
    isolation_labels = []
    
    for iso_group in consts['ISOLATION_GROUP']:
        # Get raw labels for this isolation group
        group_labels = DatasetDict.generate_labels(disp_data, [iso_group])

        # Process based on the specific group type
        if iso_group == 'gender':
            group_labels = [f"M" if x == '1' else f"F" for x in group_labels]
        elif iso_group == 'english_first_language':
            group_labels = [f"E" if x == '1' else f"NE" for x in group_labels]
        elif iso_group == 'age':
            groups = ['a15to24', 'a25to30', 'a31to35', 'a35+']
            group_labels = [groups[int(x)] for x in group_labels]
        elif iso_group == 'sbu_admit':
            groups = ['New', 'Tranf', 'Other']
            group_labels = [groups[int(x)] for x in group_labels]
        elif iso_group == 'sat_verbal_score':
            group_labels = [f"Verb{int(x)+1}" for x in group_labels]
        elif iso_group == 'sat_math_score':
            group_labels = [f"Math{int(x)+1}" for x in group_labels]
        else:
            group_labels = [f"{iso_group}{int(x)}" for x in group_labels]

        isolation_labels.append(group_labels)

    # Combine labels from all isolation groups
    # For example, if we have gender and SAT verbal, the labels will be like "M-Verb3"
    if len(isolation_labels) > 1:
        # Zip and join all label components for each data point
        combined_labels = []
        for labels in zip(*isolation_labels):
            combined_labels.append("-".join(labels))
        return combined_labels
    elif len(isolation_labels) == 1:
        # Only one isolation group
        return isolation_labels[0]
    else:
        # Fallback if no isolation groups are provided
        return ["unknown"] * len(next(iter(disp_data.values())))

def sample_model(
        model : FeatureManifoldTransformer,
        data_loader,
        consts,
        std_override: float = None,
        type: Literal['umap', 'tsne'] = 'umap',
        tag: str = ''
    ):

    # Visualize the embeddings across Gender only
    model.eval()
    with torch.no_grad():
        disp_data = next(iter(data_loader))
        disp_data = {key: value.to(DEVICE) for key, value in disp_data.items()}

        _, _, context_embed = model(disp_data, std_override)
        context_embed = context_embed.view(context_embed.size(0), -1)

        labels = get_labels(disp_data, consts)

        print(f"{consts['NAME']}{tag}")
        if type == 'tsne':
            # Generate the plots
            for i in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
                tsne_embeddings(args.outDir, context_embed.cpu().numpy(),
                                labels, title=f"{consts['NAME']}", perplexity=i)
        else:
            umap_embeddings(args.outDir, context_embed.cpu().numpy(), labels,
                            title=f"{consts['NAME']}{tag}", n_neighbors=3000,
                            min_dist=5, spread=10.0, random_state=RANDOM_SEED)


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the linear model.")
    parser.add_argument("-d", "--dataPath", type=str, help="Path to the data file.")
    parser.add_argument("-o", "--outDir", type=str, help="Path to the outputs.")
    parser.add_argument("-t", "--target", type=str, help="Target variable.")
    parser.add_argument("--tag", type=str, help="Tag to add to the output files.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--eval", action="store_true", help="Sample the model.")
    parser.add_argument("--study", action="store_true", help="Model analysis")
    parser.add_argument("-m", "--model", type=str, help="Path to the model file.")
    args = parser.parse_args()

    if args.eval and not args.model:
        raise ValueError("Must provide a model file to evaluate the model.")

    # Make output directory if it doesn't exist
    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)

    # Check if target variable is 'cins' or 'cans'
    if args.target not in ['cins', 'cans', 'toy']:
        raise ValueError("Invalid target variable. Must be 'cins' or 'cans' or 'toy'.")
    target_set = TargetSet.CINS if args.target == 'cins' else TargetSet.CANS
    print(f"Target variable: {target_set.name}")

    # Loop over D_QUESTIONS
    study_stats = {}
    for i, question in enumerate(D_QUESTIONS):
        # split the string
        DATASET, QUESTION =  question.split('_')
        QUESTION = int(QUESTION)
        print(f"Dataset: {DATASET} - Question: {QUESTION}")
        ##############################################################

        # Load the dataset and shuffle it
        set_seed(RANDOM_SEED)

        if args.target == 'toy':
            consts, model_config = get_constants(args.dataPath)
        else:
            consts, model_config = get_constants("live" ,DATASET, QUESTION)

        # Defined first to add to the model_config
        model = FeatureManifoldTransformer(model_config)

        # Create datasets
        if args.target == 'toy' and args.dataPath in ['uniform', 'linear', 'uniform-dif']:
            train_set, eval_set = generate_toy_data(consts)
            train_set = pd.DataFrame(train_set, columns=['demo1', 'demo2', 'know1', 'target'])
            eval_set = pd.DataFrame(eval_set, columns=['demo1', 'demo2', 'know1', 'target'])
            test_set = DatasetDict(eval_set, model_config)
            # Create the data loaders
            train_set = DatasetDict(train_set, model_config)
            eval_set = DatasetDict(eval_set, model_config)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            eval_loader = DataLoader(eval_set, batch_size=len(eval_set), shuffle=True)
            test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True)
            print(f"Train size: {len(train_set)}")
            print(f"Test size: {len(eval_set)}")
        else:
            data_set = load_data(args.dataPath, prep=True)
            data_set = data_set.sample(frac=1, random_state=RANDOM_SEED)
            # Data treatment
            data_set = data_set.dropna(subset=[model_config.target])
            # data_set = data_set.dropna(subset=['cins_10', 'english_first_language', 'reading_ability'])
            #Print headers
            # data_set = data_set[data_set['reading_ability'] != 0] # Low representation
            # data_set = data_set[data_set['reading_ability'] != 1] # Low representation
            # data_set['reading_ability'] = data_set['reading_ability'] - 2 # shift data
            # data_set = data_set[data_set['english_first_language'] != 2] # No information

            # Group ages into 15-24, 25-30, 31-35, 35+
            data_set = data_set.dropna(subset=['age'])
            data_set = data_set[data_set['age'] > 0]
            bins = [14.999, 24.999, 30.999, 35.999, 100]
            data_set['age'] = pd.cut(data_set['age'], bins, labels=False).astype(int)

            # Cast New, Transfer, Other -> 0, 1, 2
            data_set["sbu_admit"] = data_set["sbu_admit"].cat.codes

            # Group SAT scores into 4 groups
            data_set = data_set[data_set['sat_verbal_score'] > 0]
            data_set = data_set[data_set['sat_math_score'] > 0]

            correlation = data_set['sat_verbal_score'].corr(data_set['sat_math_score'])
            # print(f"Correlation between SAT verbal and math scores: {correlation:.4f}")

            data_set['sat_verbal_score'] = pd.qcut(data_set['sat_verbal_score'], 4, labels=False)
            data_set['sat_math_score'] = pd.qcut(data_set['sat_math_score'], 4, labels=False)
            correlation = data_set['sat_verbal_score'].corr(data_set['sat_math_score'])
            # print(f"Correlation between SAT verbal and math scores (quantiles): {correlation:.4f}")

            # data_set['sat_writing_score'] = pd.qcut(data_set['sat_writing_score'], 4, labels=False)

            # Total the CINS / CANS scores moved from can_total_pre and cin_total_pre to total_score
            if DATASET == 'cans':
                data_set['total_score'] = data_set['cans_total_pre']
            else:
                data_set['total_score'] = data_set['cins_total_pre']

            # Create the data loaders
            train_set = DatasetDict(data_set, model_config)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            print(f"Train size: {len(train_set)}")

            eval_set = DatasetDict(data_set, model_config)
            eval_loader = DataLoader(eval_set, batch_size=len(eval_set), shuffle=True)
            print(f"Eval size: {len(eval_set)}")

            # Use pd data_set to get unique tuples for demo and know
            unique_df = data_set.drop_duplicates(subset=[model_config.target,
                                                        *model_config.demos.keys(),
                                                        *model_config.knows.keys()])
            # test_set = pd.concat([unique_df] * 5, ignore_index=True)
            test_set = pd.concat([unique_df], ignore_index=True)
            test_loader = DataLoader(DatasetDict(test_set, model_config), batch_size=len(test_set), shuffle=False)

        # Define the loss function and optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.00001)

        if args.train:
            model.to(DEVICE)
            for epoch in range(MAX_EPOCHS):
                model.train()
                for i, train_run in enumerate(eval_loader):
                    train_run = {key: value.to(DEVICE) for key, value in train_run.items()}
                    optimizer.zero_grad()
                    logits, loss, _ = model(train_run)
                    loss.backward()
                    optimizer.step()

                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    for i, eval_run in enumerate(eval_loader):
                        eval_run = {key: value.to(DEVICE) for key, value in eval_run.items()}
                        logits, loss, _ = model(eval_run)

                # Early stopping
                early_stopping(loss, model)
                if early_stopping.early_stop:
                    break

                if epoch % 100 == 0:
                    print(f"Epoch: {epoch} -Loss: {loss.item()} -Patience: {early_stopping.epochs_no_improve}")
            print("Training complete.")

            # Save the model
            torch.save(model.state_dict(), os.path.join(args.outDir, f"{model_config.target}.pth"))
            print("Model saved.")

        if args.eval:
            path = os.path.join(args.outDir, f"{model_config.target}.pth")
            model.load_state_dict(torch.load(path))
            model.to(DEVICE)
            print("Model loaded.")

            # Sample the model
            for i in range(4):
                consts, model_config = get_record(DATASET, QUESTION, i)
                tag = args.tag if args.tag else ''
                sample_model(model, test_loader, consts, NOISE_STD, 'umap', tag)
