import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle

def cluster_orders(df, eps=0.3, min_samples=3):
    """Clusters orders based on geolocation using DBSCAN."""
    
    coords = df[["latitude", "longitude"]].to_numpy()
    kms_per_radian = 6371.0088  # Earth's radius in km
    eps_rad = eps / kms_per_radian  # Convert kms to radians

    clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine").fit(np.radians(coords))
    df["cluster"] = clustering.labels_
    return df

def print_clusters(df):
    """Prints clustered orders."""
    grouped = df.groupby("cluster")
    for cluster, orders in grouped:
        print(f"\nCluster {cluster}:")
        for _, order in orders.iterrows():
            print(f"  Order {order['order_id']} at ({order['latitude']}, {order['longitude']})")
