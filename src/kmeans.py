import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import preprocess_data
import os

# Fit PCA on the data
pca_n_components = 4
features_pca, _, _, _ = preprocess_data(
    test_size=0.2, random_state=1234
)

# determine the optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=500, n_init=10, random_state=0
    )
    kmeans.fit(features_pca)
    wcss.append(kmeans.inertia_)

if pca_n_components == 0:
    directory = "../results/kmeans/all_features"
else:
    directory = (
        "../results/kmeans/pca_" + str(pca_n_components) + "_principle_components"
    )
if not os.path.exists(directory):
    os.makedirs(directory)

# plot the elbow curve
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Object function value")

filename = "Elbow_Curve.png"
filepath = os.path.join(directory, filename)
plt.savefig(filepath)

plt.cla()

# perform k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=4, init="k-means++", max_iter=500, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(features_pca)

# plot the clusters
plt.scatter(
    features_pca[y_kmeans == 0, 0],
    features_pca[y_kmeans == 0, 1],
    s=10,
    c="red",
    label="Cluster 1",
)
plt.scatter(
    features_pca[y_kmeans == 1, 0],
    features_pca[y_kmeans == 1, 1],
    s=10,
    c="blue",
    label="Cluster 2",
)
plt.scatter(
    features_pca[y_kmeans == 2, 0],
    features_pca[y_kmeans == 2, 1],
    s=10,
    c="green",
    label="Cluster 3",
)
plt.scatter(
    features_pca[y_kmeans == 3, 0],
    features_pca[y_kmeans == 3, 1],
    s=10,
    c="orange",
    label="Cluster 4",
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=30,
    c="yellow",
    label="Centroids",
)
plt.title("Kmeans Clusters")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()


filename = "Kmeans_Clusters.png"
filepath = os.path.join(directory, filename)
plt.savefig(filepath)

plt.cla()
