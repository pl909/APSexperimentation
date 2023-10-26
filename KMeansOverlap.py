import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Initialize normal centers
cluster1_center = np.array([-6,8])
cluster2_center = np.array([-4,7])
cluster3_center = np.array([8,-6])
cluster4_center = np.array([4,-10])

#Initialize cluster covariance matrices
cluster1_cov = np.array([[0.4,0],[0,0.4]])
cluster2_cov = np.array([[1,2],[2,5]])
cluster3_cov = np.array([[0.4,0],[0,0.4]])
cluster4_cov = np.array([[1,2],[2,5]])

num_samples = 100

# Plot 4 clusters using Multivariate Normal
clusters1 = np.random.multivariate_normal(cluster1_center, cluster1_cov, num_samples) 
clusters2 = np.random.multivariate_normal(cluster2_center, cluster2_cov, num_samples) 
clusters3 = np.random.multivariate_normal(cluster3_center, cluster3_cov, num_samples)
clusters4 = np.random.multivariate_normal(cluster4_center, cluster4_cov, num_samples)

plotted_clusters = np.concatenate((clusters1, clusters2, clusters3, clusters4), axis = 0)

# Create Scatter Plot of Clusters

plt.scatter(plotted_clusters[:,0], plotted_clusters[:,1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Scatter Plot of 4 Clusters(Overlapping)")
plt.axis("equal")
plt.show()

# Initialize center of clusters
#Oblong initial centers

center11 = np.array([-4,0])
center12 = np.array([5,8])
center13 = np.array([1,0])
center14 = np.array([7,0])


# Store initial centers
initial_centers1 = np.vstack((center11, center12, center13, center14))


# Perform KMeans fit on first set of centers, track labels and centers
kmeans = KMeans(n_clusters=4, init=np.vstack((center11, center12, center13, center14)), n_init=1, random_state=0)
kmeans.fit(plotted_clusters)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Scatter plot of 1st set of centers
plt.scatter(plotted_clusters[:, 0], plotted_clusters[:, 1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, c='black', marker='o', label = "Final Cluster Assignments")
plt.scatter(initial_centers1[:, 0], initial_centers1[:, 1], s=100, c='grey', marker='o', label = "Initial Centers")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.title("K-Means Clustering of 4 with Overlapping Normal Centers")
plt.axis("equal")
plt.legend()
plt.show()





