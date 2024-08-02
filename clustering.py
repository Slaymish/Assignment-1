import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification, make_circles
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, AgglomerativeClustering, Birch, MeanShift
from sklearn.mixture import GaussianMixture

# Seed for reproducibility
SEED = 42

# Generate datasets
blobs, _ = make_blobs(n_samples=1000, random_state=SEED)
classification, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=SEED)
circles, _ = make_circles(n_samples=1000, noise=0.3, random_state=SEED)

datasets = [("Blobs", blobs), ("Classification", classification), ("Circles", circles)]

# Clustering algorithms
clustering_algorithms = [
    ("K-Means", lambda: KMeans(n_clusters=3, random_state=SEED)),
    ("Affinity Propagation", lambda: AffinityPropagation(random_state=SEED)),
    ("DBSCAN", lambda: DBSCAN()),
    ("Gaussian Mixture", lambda: GaussianMixture(n_components=3, random_state=SEED)),
    ("Birch", lambda: Birch(n_clusters=3)),
    ("Agglomerative Clustering", lambda: AgglomerativeClustering(n_clusters=3)),
    ("Mean Shift", lambda: MeanShift())
]

def fit_clustering_algorithms(clustering_algorithms, datasets):
    labels = []
    for _, clustering_function in clustering_algorithms:
        clusterer_labels = []
        for _, dataset in datasets:
            model = clustering_function().fit(dataset)
            if hasattr(model, "labels_"):
                cluster_predictions = model.labels_.astype(int)
            else:
                cluster_predictions = model.predict(dataset)
            clusterer_labels.append(cluster_predictions)
        labels.append(clusterer_labels)
    return labels

def build_clustering_scatterplots_table(datasets, clustering_labels):
    fig, axs = plt.subplots(len(clustering_algorithms), len(datasets), figsize=(15, 20))

    for row, (clustering_name, _) in enumerate(clustering_algorithms):
        clustering_algorithm_labels = clustering_labels[row]
        for col, (dataset_name, dataset) in enumerate(datasets):
            cell = axs[row, col]
            labels = clustering_algorithm_labels[col]

            cell.scatter(dataset[:, 0], dataset[:, 1], c=labels, s=10, cmap='viridis')
            if col == 0:
                cell.set_ylabel(clustering_name)
            if row == 0:
                cell.set_title(dataset_name)

    plt.tight_layout()
    plt.savefig("clustering_scatterplots.png", bbox_inches='tight')
    plt.show()

def analyze_results():
    analysis = """
    Comparison and Analysis of Clustering Algorithms on Datasets:

    1. K-Means:
       - Performs well on spherical clusters.
       - May struggle with clusters of varying density and non-linear shapes.

    2. Affinity Propagation:
       - Identifies exemplars rather than centroids.
       - Handles non-convex shapes well but can be sensitive to preference parameter.

    3. DBSCAN:
       - Effective for arbitrary-shaped clusters.
       - Requires careful tuning of eps and min_samples.

    4. Gaussian Mixture Model:
       - Assumes Gaussian distributions for clusters.
       - Works well on spherical clusters, less effective on non-linear structures.

    5. BIRCH:
       - Handles large datasets efficiently.
       - Performs well on spherical clusters but may struggle with non-linear clusters.

    6. Agglomerative Clustering:
       - Builds a hierarchy of clusters.
       - Effective for various shapes, but computationally intensive.

    7. Mean Shift:
       - Identifies dense regions as clusters.
       - Effective for various shapes but can be computationally intensive.
    """
    print(analysis)

def main():
    clustering_labels = fit_clustering_algorithms(clustering_algorithms, datasets)
    build_clustering_scatterplots_table(datasets, clustering_labels)
    analyze_results()

if __name__ == "__main__":
    main()