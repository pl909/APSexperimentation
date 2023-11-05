from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load the image
image_path = '1985-mike-tyson-001338270jpg.jpg'  # Replace with your image path
original_image = Image.open(image_path)

# Resize the image (too large)
original_image = original_image.resize((original_image.width // 2, original_image.height // 2))

# Convert the image to a numpy array
pixels = np.array(original_image)
pixels_reshaped = pixels.reshape(-1, 3)

n_clusters = 9  
gmm = GaussianMixture(n_components=n_clusters, random_state=0)
gmm.fit(pixels_reshaped)

# Predict the cluster for each pixel and get means
clusters = gmm.predict(pixels_reshaped)
mean_colors = gmm.means_.astype(int)

# Function to handle ties in responsibilities
def assign_cluster_with_ties(responsibilities):
    max_responsibility = responsibilities.max()
    tied_clusters = np.flatnonzero(responsibilities == max_responsibility)
    return np.random.choice(tied_clusters)

#assignment to each cluster
assignments = np.array([assign_cluster_with_ties(r) for r in gmm.predict_proba(pixels_reshaped)])

# Map each pixel and reshape
pop_art_pixels = np.array([mean_colors[cluster] for cluster in assignments])
pop_art_image = pop_art_pixels.reshape(original_image.size[1], original_image.size[0], 3)
pop_art_image = Image.fromarray(np.uint8(pop_art_image))


# Plot the original and the Pop Art images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(pop_art_image)
axes[1].set_title('Pop Art Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
