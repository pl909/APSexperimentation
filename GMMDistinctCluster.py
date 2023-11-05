import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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
plt.title("Scatter Plot of 4 Clusters")
plt.axis("equal")
plt.show()

# Fit Gaussian Mixture model to clusters.
gmm = GaussianMixture(n_components=4, covariance_type = 'full')
gmm.fit(plotted_clusters)
responsibilities = gmm.predict_proba(plotted_clusters)


# Function to convert CMYK to RGB
def cmyk_to_rgb(c, m, y, k):
    r = (1.0-c) * (1.0-k)
    g = (1.0-m) * (1.0-k)
    b = (1.0-y) * (1.0-k)
    return r, g, b


rgb_colors = np.zeros((plotted_clusters.shape[0], 3))

# Calculate the RGB values from the responsibilities
for i, responsibility in enumerate(responsibilities):
    c, m, y, k = responsibility
    rgb_colors[i] = cmyk_to_rgb(c, m, y, k)

# Create the scatter plot with the calculated RGB values
plt.figure(figsize=(8, 6))
plt.scatter(plotted_clusters[:, 0], plotted_clusters[:, 1], c=rgb_colors, alpha=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Scatter Plot with GMM Component Responsibilities")
plt.axis("equal")
plt.show()


# Oblong inital centers Kmeans
initial_centers = np.array([[-4, 0], [5, 8], [3, 4], [7, 0]])

# Run K-Means with the initial centers
kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=1, random_state=0)
kmeans.fit(plotted_clusters)

# Initialize the GMM with the KMeans
gmm = GaussianMixture(n_components=4, covariance_type='full', means_init=kmeans.cluster_centers_)
gmm.fit(plotted_clusters)

# Points to evaluate GMM PDF
x = np.linspace(plotted_clusters[:, 0].min() - 1, plotted_clusters[:, 0].max() + 1, num=100)
y = np.linspace(plotted_clusters[:, 1].min() - 1, plotted_clusters[:, 1].max() + 1, num=100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

# Compute the GMM PDF for each point on the grid
Z = gmm.score_samples(XX)
Z = Z.reshape(X.shape)

# Plot the contour plot for the GMM PDF
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=15, linewidths=1.5)
plt.scatter(plotted_clusters[:, 0], plotted_clusters[:, 1], c='grey', alpha=0.5)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x')  # Plot the GMM means
plt.title('Contour Plot for the GMM PDF w/ KMeans 3a Initialized Centers')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()




# Distinct Initial Centers KMeans
initial_centers = np.array([[-7, 7], [-7, -7], [7, 7], [7, -7]])

# Run K-Means with the initial centers
kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=1, random_state=0)
kmeans.fit(plotted_clusters)

# Initialize the GMM with the KMeans
gmm = GaussianMixture(n_components=4, covariance_type='full')
gmm.fit(plotted_clusters)

# Points to evaluate GMM PDF
x = np.linspace(plotted_clusters[:, 0].min() - 1, plotted_clusters[:, 0].max() + 1, num=100)
y = np.linspace(plotted_clusters[:, 1].min() - 1, plotted_clusters[:, 1].max() + 1, num=100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

# Compute the GMM PDF for each point on the grid
Z = gmm.score_samples(XX)
Z = Z.reshape(X.shape)

# Plot the contour plot for the GMM PDF
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=15, linewidths=1.5)
plt.scatter(plotted_clusters[:, 0], plotted_clusters[:, 1], c='grey', alpha=0.5)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x')  # Plot the GMM means
plt.title('Contour Plot for the GMM PDF w/ KMeans 3b Initialized Centers')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()


# Do Expectation Maximization

# Initialize the GMM with the KMeans
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state = 0)
gmm.fit(plotted_clusters)

# Points to evaluate GMM PDF
x = np.linspace(plotted_clusters[:, 0].min() - 1, plotted_clusters[:, 0].max() + 1, num=100)
y = np.linspace(plotted_clusters[:, 1].min() - 1, plotted_clusters[:, 1].max() + 1, num=100)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

# Compute the GMM PDF for each point on the grid
Z = gmm.score_samples(XX)
Z = Z.reshape(X.shape)

# Plot the contour plot for the GMM PDF
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=15, linewidths=1.5)
plt.scatter(plotted_clusters[:, 0], plotted_clusters[:, 1], c='grey', alpha=0.5)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x')  # Plot the GMM means
plt.title('Contour Plot for the GMM PDF using Expectation-Maximization')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()
