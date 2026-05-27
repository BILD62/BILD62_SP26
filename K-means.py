import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs

colors = ["#963d5a", "#6279b8", "#6DA34D", "#496f5d", "#e7bb41"]
colors = np.array(colors)

n_samples = 100
n_components = 3
X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.6, random_state=0)

# Perform the k-means clustering
n_iterations = 15

centroid_color = "#0c120c"

fig, ax = plt.subplots(figsize=(8,6))

# Change seed - this will give you different initial centroids
new_seed = 1
np.random.seed(new_seed)

# Define the update function to update the frame
current_centroids = np.random.randn(n_components, 2)

def update(frame):
    global current_centroids
    ax.clear()

    if frame < 2:
        ax.scatter(X[:,0], X[:,1], color='k', s=120, alpha=0.75)
    else:
        # Perform 1 step of K-means clustering
        kmeans = KMeans(
            n_clusters=n_components,
            random_state=0,
            max_iter=1,
            init=current_centroids,
            n_init=1).fit(X)
        # Update centroid for each iteration
        current_centroids = kmeans.cluster_centers_

        ax.scatter(X[:,0], X[:,1], color=colors[kmeans.labels_], s=120, alpha=0.75)
        ax.scatter(current_centroids[:,0], current_centroids[:,1],
                   c=centroid_color, marker='X', s=200)
        ax.set_title(f'Iteration {frame-1}')

# Define animation object
animation = FuncAnimation(fig, update, frames=n_iterations+2, repeat=False)

# Close the figure to prevent it from displaying as a static plot
plt.close(fig)

# Display the animation interactively in the notebook
from IPython.display import HTML, display
display(HTML(animation.to_jshtml()))