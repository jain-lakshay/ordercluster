import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the src/ folder is included in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now import from src.cluster
from cluster import cluster_orders, print_clusters

# Sample data (You can replace this with actual data from `data/orders.csv`)
data = {
    "order_id": [1, 2, 3, 4, 5],
    "latitude": [28.7041, 28.7042, 28.705, 28.7005, 28.7043],
    "longitude": [77.1025, 77.1026, 77.1035, 77.1, 77.1027],
}

df = pd.DataFrame(data)

# Cluster the orders
df_clustered = cluster_orders(df, eps=0.2, min_samples=2)  # Try different values


# Print clusters
print_clusters(df_clustered)
# Plot the clustered orders
plt.figure(figsize=(8,6))
plt.scatter(df["longitude"], df["latitude"], c=df_clustered["cluster"], cmap="viridis", marker="o", edgecolors="k")

# Labels & Customization
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Order Clustering Visualization")
plt.colorbar(label="Cluster ID")
plt.grid(True)

# Show plot
plt.show()