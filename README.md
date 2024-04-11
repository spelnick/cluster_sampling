# cluster_sampling

Random sampling is bad!! Here's an alternative: cluster your data before extracting train and test sets. This repo contains functions that can help you cluster your data quickly (including identifying best number of clusters) and then sample more evenly from each cluster.

How it works:
 - Use scikit-learn to cluster
 - For each observation, the distance between it and its respective cluster center is calculated.
 - The training set is then selected by going cluster to cluster, selecting the observation closest to the center, selecting the observation farthest from the center, and then random sampling any remaining observations needed to reach user's defined training set size.
 - The test set is then randomly selected each cluster removing any observations already selected for the training set.
 - There are 2 exceptions to this process when iterating over each cluster:
    1. If a cluster has only 1 observation, the train set is skipped for this cluster and the observation is added to the test set.
    2. If a cluster has only 2 observations, the observation closest to the center is added to the train set and the other is added to the test set.
