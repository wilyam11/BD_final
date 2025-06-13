# main.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("public_data.csv")

# Separate features and ids
ids = df["id"]
X = df.drop(columns=["id"]).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering with KMeans
k = 4 * X.shape[1] - 1
model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

# Create submission file
submission = pd.DataFrame({"id": ids, "label": labels})
submission.to_csv("public_submission.csv", index=False)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab20', s=30, alpha=0.7)
plt.title("KMeans Clustering (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster Label")
plt.grid(True)
plt.tight_layout()
plt.savefig("clusters_plot.png")
plt.show()
