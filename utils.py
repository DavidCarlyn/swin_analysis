import time

from sklearn.manifold import TSNE

def run_TSNE(features):
    tsne = TSNE(n_components=2)
    print(f"Running TSNE with {len(features)} data points")
    time_start = time.time()
    tsne_features = tsne.fit_transform(features)
    print(f"Completed TSNE in {time.time() - time_start}")
    return tsne_features