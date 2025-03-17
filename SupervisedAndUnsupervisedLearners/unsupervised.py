import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def optimal_gmm(X, max_components=5):
    #Finds the optimal GMM with the lowest BIC and returns the model along with BIC scores
    lowest_bic = np.inf
    best_gmm = None
    bic_scores = []
    n_components_range = range(1, max_components + 1)
    for n_components in n_components_range:
        print("testing" + " " + str(n_components))
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
    return best_gmm, bic_scores, list(n_components_range)

def plot_bic_scores(n_components_range, bic_scores, dataset_label):
    #Plots the BIC scores for different numbers of clusters
    plt.figure()
    plt.plot(n_components_range, bic_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('BIC Score')
    plt.title(f'BIC Scores for {dataset_label}')
    plt.xticks(n_components_range)
    plt.tight_layout()
    filename = f'{dataset_label}_BIC.png'
    plt.savefig(filename)
    print(f"BIC plot saved as {filename}")
    plt.show()

def plot_clusters(X, gmm, dataset_label):
    #Plots the data points colored by the cluster labels along with annotated cluster numbers at the centers.
    labels = gmm.predict(X)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolor='k')
    plt.title(f'Clusters for {dataset_label}')
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 2')
    centers = gmm.means_
    for i, center in enumerate(centers):
        plt.text(center[0], center[1], str(i), fontsize=12, color='red', fontweight='bold',
                 ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    filename = f'{dataset_label}_clusters.png'
    plt.savefig(filename)
    print(f"Cluster plot saved as {filename}")
    plt.show()

def process_dataset(file_path, dataset_label):
    #Fully processes a dataset
    data = pd.read_csv(file_path, header=None)
    X = data.values
    best_gmm, bic_scores, n_components_range = optimal_gmm(X, max_components=5)
    print(f"\nOptimal number of clusters for {dataset_label}: {best_gmm.n_components}")
    for i in range(best_gmm.n_components):
        print(f"\nCluster {i}:")
        print("Mean:", best_gmm.means_[i])
        print("Covariance:\n", best_gmm.covariances_[i])
    plot_bic_scores(n_components_range, bic_scores, dataset_label)
    plot_clusters(X, best_gmm, dataset_label)
    return best_gmm

if __name__ == '__main__':
    np.random.seed(hash("Actually, ChatGPT CAN'T have my job") % (2**32))
    print("Processing dataset A:")
    gmm_A = process_dataset('hw2-Part2-datasetA.csv', 'DatasetA')  
    print("\nProcessing dataset B:")
    gmm_B = process_dataset('hw2-Part2-datasetB.csv', 'DatasetB')  
