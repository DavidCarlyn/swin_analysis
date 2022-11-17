import time
import json

import numpy as np
import matplotlib.pyplot as plt

from options import RESULTS_DIR, NUM_LEVELS, LABEL_MAP_PATH
from utils import run_TSNE


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

def sample_data(features, labels, num_samples_per_class=10):
    """
    Samples data from top two levels of the hierarchy
    """
    sam_features = []
    sam_labels = []
    num_classes = max(labels[:, 1]) + 1 # Number of classes on the 2nd level of the hierarchy
    for cls in range(num_classes):
        idx = labels[:, 1] == cls
        feat_idx = np.random.choice(features[idx].shape[0], num_samples_per_class, replace=False)
        feat = features[idx][feat_idx].tolist()
        sam_features.extend(feat)
        sam_labels.extend(labels[idx][feat_idx].tolist())
    
    sam_features = np.array(sam_features)
    sam_labels = np.array(sam_labels)
    print(sam_features.shape)
    print(sam_labels.shape)

    return sam_features, sam_labels

def visualize(features, labels, label_map, subsample=False, rerun_TSNE=False, top_k=6):
    if rerun_TSNE:
        tsne_features = features
    else:
        tsne_features = run_TSNE(features)

    data_queue = [(tsne_features, labels, 0)]
    while len(data_queue) > 0:
        feats, lbls, lvl = data_queue.pop(0)
        if rerun_TSNE:
            tsne_feats = run_TSNE(feats)
        else:
            tsne_feats = feats
        lvl_lbl_map = label_map[lvl]
        parent_name = None
        if lvl > 0:
            parent_name = "_".join(lvl_lbl_map[str(lbls[0, lvl])].split("_")[:-1])
            print(f"Plotting Level {lvl}: {parent_name}")
        else:
            print(f"Plotting Level 0: Root")

        lbl_lengths = []
        for lbl in set(lbls[:, lvl]):
            idx = lbls[:, lvl] == lbl
            lbl_lengths.append([lbl, len(tsne_feats[idx])])
        lbl_lengths = sorted(lbl_lengths, key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(20, 6))
        plt.axis('off')
        most_feats = None
        most_lbls = None
        highest_num = 0
        for lbl in set(lbls[:, lvl]):
            if lbl not in np.array(lbl_lengths)[:top_k, 0]: continue
            idx = lbls[:, lvl] == lbl
            feat = tsne_feats[idx]
            name = lvl_lbl_map[str(lbl)].split("_")[-1]
            plt.scatter(feat[:, 0], feat[:, 1], label=name)
            if len(feat) > highest_num:
                highest_num = len(feat)
                if rerun_TSNE:
                    most_feats = feats[idx]
                else:
                    most_feats = feat
                most_lbls = lbls[idx]

        # Add to queue
        if (lvl+1) < NUM_LEVELS:
            data_queue.append((most_feats, most_lbls, lvl+1))

        plt.legend()
        if lvl > 0:
            #plt.title(f"Depth {lvl}: {parent_name}")
            plt.savefig(f"output/depth_{lvl}_{parent_name}.png")
        else:
            #plt.title(f"Depth {lvl}")
            plt.savefig(f"output/depth_{lvl}.png")

        plt.close()

#    for lvl, lvl_map in enumerate(label_map):
#        for lbl in range(max(labels[:, lvl])+1):
#            idx = labels[:, lvl] == lbl
#            feat = tsne_features[idx]
#            plt.scatter(feat[:, 0], feat[:, 1], label=lbl)

def load_label_map():
    with open(LABEL_MAP_PATH, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    np.random.seed(2022)
    data_type = "val"
    model_type = "non_hierarchy"

    features, logits, labels = load_data(data_type, model_type)
    #sam_features, sam_labels = sample_data(features, labels, num_samples_per_class=40)

    label_map = load_label_map()

    visualize(features, labels, label_map, rerun_TSNE=True)