import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def extract_features(image_dir, resize = True):
    """Get pixel features for each class of images
    
    Inputs:
    - image_dir: Path to directory to images
    Outputs:
    - features: Array of shape num_images x num_features
    """
    features = []
    images = os.listdir(image_dir)
    for image in images:
        img_path = os.path.join(image_dir, image)
        img = cv2.imread(img_path)
        img = img.astype('float32') / 255.0
        if resize:
            img_smaller = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            features.append(img_smaller.flatten())
        else:
            features.append(img.flatten())

    return np.array(features)



# Laplacian Eigenmap Embeddings
def laplacian_eigenmap(features, n_neighbors=6, n_components=3, weights = 'gaussian') :
    """Computes and plots laplacian eigenmap embeddings
    Inputs:
    - features: N x d array of samples x features
    - n_neighbors: Neighbors for graph construction
    - n_components: Number of embedding dimensions
    - weights: Weighting scheme for choosing w_ij

    Outputs:
    - embedding: N x n_components array of 2D embeddings"""
    if weights == 'gaussian':
        # Weight Matrix
        knn_graph = kneighbors_graph(features, n_neighbors=n_neighbors, mode='distance', include_self=False)
        # sigma2 = max( np.median(knn_graph.data) ** 2, 1e-12)
        sigma2 = 1
        knn_graph.data = np.exp(-knn_graph.data ** 2 / (2 * sigma2))
    else:
        # Adjacency Matrix
        knn_graph = kneighbors_graph(features, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    
    knn_graph = (knn_graph + knn_graph.T) / 2

    # Normalized Laplacian Matrix
    laplacian = csgraph.laplacian(knn_graph, normed=True)

    eigenvalues, eigenvectors = eigsh(laplacian, k=n_components+1, which='SM')

    assert np.isclose(eigenvalues[0], 0)

    embedding = eigenvectors[:, 1:n_components+1]
    return embedding


def main(vis_embeddings = False):
    """Main method"""

    img = cv2.imread('C:\\Users\\jluo1\\Documents\\repos\\histopathology_classification\\data\\HMU-GC-HE-30K\\all_image\\MUS\\MUS_6.png')
    img = img.astype('float32') / 255.0
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Original Image of Size {img.shape}')
    plt.show()


    nor_features = extract_features('C:\\Users\\jluo1\\Documents\\repos\\histopathology_classification\\data\\HMU-GC-HE-30K\\all_image\\NOR')
    adi_features = extract_features('C:\\Users\\jluo1\\Documents\\repos\\histopathology_classification\\data\\HMU-GC-HE-30K\\all_image\\ADI')
    #tum_features = extract_features('C:\\Users\\jluo1\\Documents\\repos\\histopathology_classification\\data\\HMU-GC-HE-30K\\all_image\\TUM')

    all_features = np.concatenate((nor_features, adi_features), axis=0)
    embedding = laplacian_eigenmap(all_features, n_neighbors=10, n_components=3, weights='gaussian')
    embeddings_df = pd.DataFrame(embedding, columns=[f'Dim{i+1}' for i in range(embedding.shape[1])])
    embeddings_df['Label'] = ['NOR'] * len(nor_features) + ['ADI'] * len(adi_features)
    #embeddings_df.to_csv('C:\\Users\\jluo1\\Documents\\repos\\histopathology_classification\\laplacian_eigenmap_embeddings.csv', index=False)

    if vis_embeddings:
        fig = plt.figure(figsize=(6, 6))

        # Plot 3D Scatter Plot of Embeddings
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=['red'] * len(nor_features) + ['blue'] * len(adi_features), alpha=0.7)
        plt.title('Laplacian Eigenmap 3D Projection of Histopathology Images')
        plt.savefig(os.path.join('C:\\Users\\jluo1\\Documents\\repos\\histopathology_classification\\images', 'laplacian_eigenmap_3d.png'))
        plt.show()

        # Plot 2D Scatter Plot of Embeddings
        plt.figure(figsize=(6, 6))
        plt.scatter(embedding[:, 1], embedding[:, 2], c=['red'] * len(nor_features) + ['blue'] * len(adi_features), alpha=0.7)
        plt.title('Laplacian Eigenmap 2D Projection of Histopathology Images')
        plt.savefig(os.path.join('C:\\Users\\jluo1\\Documents\\repos\\histopathology_classification\\images', 'laplacian_eigenmap_2d.png'))
        plt.show()


if __name__ == "__main__":
    main(vis_embeddings=True)