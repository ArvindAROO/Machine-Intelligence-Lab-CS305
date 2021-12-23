import numpy as np


class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self
    def assign_cluster(self, point):
        """
        Assign a cluster to a point
        Args:
            point: A single point (1 x D) vector (numpy float)
        Returns:
            cluster_id: Cluster ID (int)
        """
        distances = np.linalg.norm(point - self.centroids, axis=1)
        return np.argmin(distances)

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        cluster_assign = np.zeros(data.shape[0], dtype=int)

        for idx, point in enumerate(data):
            cluster_assign[idx] = self.assign_cluster(point)

        return cluster_assign
        # done

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
        Change self.centroids
        """
        for cluster_id in range(self.n_cluster):
            cluster_points = data[cluster_assgn == cluster_id]
            self.centroids[cluster_id] = np.mean(cluster_points, axis=0)
        # done

        
    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
        Returns:
            metric : Summation of number of clusters, summation of number of cases square of difference between centroid of cluster j and ith case of x
        """ 
        # Initialize the distance array
        distance = np.zeros((data.shape[0]))
        # SSE
        for k in range(self.n_cluster):
            # Calucluate error
            squared = np.square(data[cluster_assign==k] - self.centroids[k])
            distance[cluster_assign == k] = np.sqrt(np.sum(squared, axis=1))

        # Sum and Square errors
        return np.sum(np.square(distance))
        