"""
Utility functions for the SOM-VAE model
Copyright (c) 2018
Author: Vincent Fortuin
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

import numpy as np
from scipy.special import comb
from sklearn.utils.linear_assignment_ import linear_assignment


def interpolate_arrays(arr1, arr2, num_steps=100, interpolation_length=0.3):
    """Interpolates linearly between two arrays over a given number of steps.
    The actual interpolation happens only across a fraction of those steps.

    Args:
        arr1 (np.array): The starting array for the interpolation.
        arr2 (np.array): The end array for the interpolation.
        num_steps (int): The length of the interpolation array along the newly created axis (default: 100).
        interpolation_length (float): The fraction of the steps across which the actual interpolation happens (default: 0.3).

    Returns:
        np.array: The final interpolated array of shape ([num_steps] + arr1.shape).
    """
    assert arr1.shape == arr2.shape, "The two arrays have to be of the same shape"
    start_steps = int(num_steps*interpolation_length)
    inter_steps = int(num_steps*((1-interpolation_length)/2))
    end_steps = num_steps - start_steps - inter_steps
    interpolation = np.zeros([inter_steps]+list(arr1.shape))
    arr_diff = arr2 - arr1
    for i in range(inter_steps):
        interpolation[i] = arr1 + (i/(inter_steps-1))*arr_diff
    start_arrays = np.concatenate([np.expand_dims(arr1, 0)] * start_steps)
    end_arrays = np.concatenate([np.expand_dims(arr2, 0)] * end_steps)
    final_array = np.concatenate((start_arrays, interpolation, end_arrays))
    return final_array


def compute_NMI(cluster_assignments, class_assignments):
    """Computes the Normalized Mutual Information between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    
    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The NMI value.
    """
    assert len(cluster_assignments) == len(class_assignments), "The inputs have to be of the same length."
    
    clusters = np.unique(cluster_assignments)
    classes = np.unique(class_assignments)
    
    num_samples = len(cluster_assignments)
    num_clusters = len(clusters)
    num_classes = len(classes)
    
    assert num_classes > 1, "There should be more than one class."
        
    cluster_class_counts = {cluster_: {class_: 0 for class_ in classes} for cluster_ in clusters}
    
    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1
    
    cluster_sizes = {cluster_: sum(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()}
    class_sizes = {class_: sum([cluster_class_counts[clus][class_] for clus in clusters]) for class_ in classes}
    
    I_cluster_class = H_cluster = H_class = 0
    
    for cluster_ in clusters:
        for class_ in classes:
            if cluster_class_counts[cluster_][class_] == 0:
                pass
            else:
                I_cluster_class += (cluster_class_counts[cluster_][class_]/num_samples) * \
                (np.log((cluster_class_counts[cluster_][class_]*num_samples)/ \
                        (cluster_sizes[cluster_]*class_sizes[class_])))
                        
    for cluster_ in clusters:
        H_cluster -= (cluster_sizes[cluster_]/num_samples) * np.log(cluster_sizes[cluster_]/num_samples)
                
    for class_ in classes:
        H_class -= (class_sizes[class_]/num_samples) * np.log(class_sizes[class_]/num_samples)
        
    NMI = (2*I_cluster_class)/(H_cluster+H_class)
    
    return NMI


def compute_purity(cluster_assignments, class_assignments):
    """Computes the purity between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    
    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The purity value.
    """
    assert len(cluster_assignments) == len(class_assignments)
    
    num_samples = len(cluster_assignments)
    num_clusters = len(np.unique(cluster_assignments))
    num_classes = len(np.unique(class_assignments))
    
    cluster_class_counts = {cluster_: {class_: 0 for class_ in np.unique(class_assignments)}
                            for cluster_ in np.unique(cluster_assignments)}
    
    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1
        
    total_intersection = sum([max(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()])
    
    purity = total_intersection/num_samples
    
    return purity


def compute_AMI(cluster_assignments, class_assignments):
    """Computes the Adjusted Mutual Information between cluster and class assignments.

    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.

    Returns:
        float: The AMI value.
    """
    assert len(cluster_assignments) == len(class_assignments), "The inputs have to be of the same length."

    clusters = np.unique(cluster_assignments)
    classes = np.unique(class_assignments)

    num_samples = len(cluster_assignments)
    num_clusters = len(clusters)
    num_classes = len(classes)

    assert num_classes > 1, "There should be more than one class."

    cluster_class_counts = {cluster_: {class_: 0 for class_ in classes} for cluster_ in clusters}

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1

    cluster_sizes = {cluster_: sum(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()}
    class_sizes = {class_: sum([cluster_class_counts[clus][class_] for clus in clusters]) for class_ in classes}

    mutual_info = 0
    for cluster_ in clusters:
        for class_ in classes:
            if cluster_class_counts[cluster_][class_] > 0:
                mutual_info += cluster_class_counts[cluster_][class_] / num_samples * \
                               np.log((cluster_class_counts[cluster_][class_] * num_samples) /
                                      (cluster_sizes[cluster_] * class_sizes[class_]))

    cluster_entropy = -sum(
        [(cluster_sizes[cluster_] / num_samples) * np.log(cluster_sizes[cluster_] / num_samples) for cluster_ in
         clusters])
    class_entropy = -sum(
        [(class_sizes[class_] / num_samples) * np.log(class_sizes[class_] / num_samples) for class_ in classes])

    expected_mutual_info = 0
    for cluster_size in cluster_sizes.values():
        for class_size in class_sizes.values():
            for n_ij in range(max(1, cluster_size + class_size - num_samples), min(cluster_size, class_size) + 1):
                expected_mutual_info += (n_ij / num_samples) * np.log(
                    (n_ij * num_samples) / (cluster_size * class_size)) * \
                                        (comb(cluster_size, n_ij) * comb(num_samples - cluster_size,
                                                                         class_size - n_ij) / comb(num_samples,
                                                                                                   class_size))

    ami = (mutual_info - expected_mutual_info) / (max(cluster_entropy, class_entropy) - expected_mutual_info)

    return ami


def compute_silhouette_score(data, cluster_assignments):
    """Computes the Silhouette Score for the clustering.

    Args:
        data (np.array): The data points.
        cluster_assignments (list): List of cluster assignments for every point.

    Returns:
        float: The Silhouette Score.
    """
    num_clusters = len(np.unique(cluster_assignments))
    if num_clusters == 1 or num_clusters == len(cluster_assignments):
        return 0

    num_samples = len(data)
    silhouette_scores = np.zeros(num_samples)

    for i in range(num_samples):
        own_cluster = cluster_assignments[i]
        own_cluster_points = data[cluster_assignments == own_cluster]
        other_cluster_points = data[cluster_assignments != own_cluster]

        a_i = np.mean(np.linalg.norm(own_cluster_points - data[i], axis=1))
        b_i = np.min([np.mean(
            np.linalg.norm(data[cluster_assignments == c] - data[i], axis=1)) for c in np.unique(cluster_assignments) if
            c != own_cluster])

        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(silhouette_scores)


def compute_calinski_harabasz_score(data, cluster_assignments):
    """Computes the Calinski-Harabasz Index for the clustering.

    Args:
        data (np.array): The data points.
        cluster_assignments (list): List of cluster assignments for every point.

    Returns:
        float: The Calinski-Harabasz Index.
    """
    num_clusters = len(np.unique(cluster_assignments))
    num_samples = len(data)

    overall_mean = np.mean(data, axis=0)
    cluster_means = {c: np.mean(data[cluster_assignments == c], axis=0) for c in np.unique(cluster_assignments)}
    cluster_sizes = {c: np.sum(cluster_assignments == c) for c in np.unique(cluster_assignments)}

    between_cluster_dispersion = sum(
        [cluster_sizes[c] * np.linalg.norm(cluster_means[c] - overall_mean) ** 2 for c in cluster_means])
    within_cluster_dispersion = sum(
        [np.sum(np.linalg.norm(data[cluster_assignments == c] - cluster_means[c], axis=1) ** 2) for c in cluster_means])

    return (between_cluster_dispersion / (num_clusters - 1)) / (within_cluster_dispersion / (num_samples - num_clusters))


def compute_davies_bouldin_score(data, cluster_assignments):
    """Computes the Davies-Bouldin Index for the clustering.

    Args:
        data (np.array): The data points.
        cluster_assignments (list): List of cluster assignments for every point.

    Returns:
        float: The Davies-Bouldin Index.
    """
    num_clusters = len(np.unique(cluster_assignments))
    cluster_means = {c: np.mean(data[cluster_assignments == c], axis=0) for c in np.unique(cluster_assignments)}
    cluster_sizes = {c: np.sum(cluster_assignments == c) for c in np.unique(cluster_assignments)}

    s = {c: np.mean(np.linalg.norm(data[cluster_assignments == c] - cluster_means[c], axis=1)) for c in cluster_means}

    db_index = 0
    for i in cluster_means:
        max_ratio = 0
        for j in cluster_means:
            if i != j:
                ratio = (s[i] + s[j]) / np.linalg.norm(cluster_means[i] - cluster_means[j])
                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio

    return db_index / num_clusters
