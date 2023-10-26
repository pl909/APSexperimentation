from PIL import Image
import cv2
from sklearn.cluster import KMeans
import numpy as np

# Initializing image and kmeans
image = Image.open('1985-mike-tyson-001338270jpg.jpg')
data = np.array(image)
if data is None:
    print("uh")
print(data.shape)
data_2d = data.reshape(-1, 3)
kmeans = KMeans(n_clusters=9, random_state=0).fit(data_2d)  # or data_hsv for HSV

# Getting cluster Centers
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_
new_data = cluster_centers[labels].astype(int)

np.random.seed(10010)  # Optional: for reproducible random colors
random_colors = np.random.randint(0, 256, size=(kmeans.n_clusters, 3))  # Generating random colors
new_data = random_colors[labels]  # Assigning random colors based on cluster labels

#reshaping and displaying
new_data = new_data.reshape(data.shape)
new_image = Image.fromarray(new_data.astype('uint8'))
original_image = image

# Displaying side by side
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(new_image)
plt.title('Pop Art')

plt.show()