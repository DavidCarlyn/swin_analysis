import time
import json
from tqdm import tqdm

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from options import RESULTS_DIR, NUM_LEVELS, LABEL_MAP_PATH


def load_data(data_type, model_type):
    if model_type == "hierarchy":
        features = np.load(f'{RESULTS_DIR}/{data_type}_features.npz')['features']
        labels = np.load(f'{RESULTS_DIR}/{data_type}_labels.npz')['labels']
        logits = []
        for lvl in range(NUM_LEVELS):
            logits.append(np.load(f'{RESULTS_DIR}/{data_type}_logits_lvl_{lvl}.npz')['logits'])
    else:
        features = np.load(f'{RESULTS_DIR}/{data_type}_{model_type}_features.npz')['features']
        labels = np.load(f'{RESULTS_DIR}/{data_type}_{model_type}_labels.npz')['labels']
        logits = np.load(f'{RESULTS_DIR}/{data_type}_{model_type}_logits.npz')['logits']

    return features, logits, labels
    
def load_label_map():
    with open(LABEL_MAP_PATH, 'r') as f:
        return json.load(f)

def calc_rank_matrix(dist_matrix):
    N = len(dist_matrix)
    rank_matrix = np.zeros((N, N))
    for i in tqdm(range(N), desc="Calculating rank matrix"):
        last_tie = -1
        rank_accum = 0
        sorted_dist = dist_matrix[i, np.argsort(dist_matrix[i])]
        for j in range(N):
            if j != 0 and sorted_dist[j] == sorted_dist[j-1]:
                rank_accum += j+1
                if last_tie == -1:
                    last_tie = j-1
                    rank_accum += j
            else:
                rank_matrix[i, j] = j+1
                if last_tie != -1:
                    rank_matrix[i, last_tie:j] = rank_accum / (j - last_tie)
                    rank_accum = 0
                    last_tie = -1
        if last_tie != -1:
            rank_matrix[i, last_tie:] = rank_accum / (N - last_tie)
    return rank_matrix

def calc_true_rank_matrix(labels):
    N = len(labels)
    dist_matrix = np.zeros((N, N))
    for i, x in tqdm(enumerate(labels), desc="Calculating hierarchical distances"):
        diffs = 2*(x - labels).sum(1)
        dist_matrix[i, :] = diffs

    return calc_rank_matrix(dist_matrix)

def calc_feat_rank_matrix(features):
    N, D = features.shape
    dist_matrix = np.zeros((N, N))
    for i, x in tqdm(enumerate(features), desc="Calculating feature distances"):
        dist = np.sqrt(((x - features)**2).sum(1))
        dist_matrix[i, :] = dist

    return calc_rank_matrix(dist_matrix)

def calc_spearman_rank_correlation(A, B):
    N = len(A)
    spearman_correlation =  1 - ((6*((A - B)**2).sum(1)) / (N*((N**2)-1)))
    return spearman_correlation.mean()


if __name__ == "__main__":
    np.random.seed(2022)
    data_type = "val"
    model_type = "hierarchy"

    features, logits, labels = load_data(data_type, model_type)
    label_map = load_label_map()

    print("Calculating feature rank matrix")
    feat_rank_matrix = calc_feat_rank_matrix(features)
    print("Calculating true rank matrix")
    true_rank_matrix = calc_true_rank_matrix(labels)

    print("Calculating Spearman rank correlation")
    corr = calc_spearman_rank_correlation(true_rank_matrix, feat_rank_matrix)
    print(corr)