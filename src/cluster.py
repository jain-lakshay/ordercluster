import pandas as pd
import spacy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from fuzzywuzzy import process
from geopy.geocoders import Nominatim

# Load NLP model
nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="order_clustering")

def extract_locality(address):
    """Extracts locality (like sector, block, or area name) from an address using NLP."""
    doc = nlp(address)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text.lower()
    return "unknown"

def cluster_orders(df, eps=0.3, min_samples=2):
    """Clusters orders based on locality and geolocation."""
    
    # Apply NLP to extract locality names
    df["locality"] = df["address"].apply(lambda x: extract_locality(x) if pd.notna(x) else "unknown")

    # Convert locality names into numeric values for clustering
    unique_localities = df["locality"].unique()
    locality_map = {loc: i for i, loc in enumerate(unique_localities)}
    df["locality_id"] = df["locality"].map(locality_map)

    # Apply DBSCAN on geolocation and locality ID
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    df["cluster"] = clustering_model.fit_predict(df[["latitude", "longitude", "locality_id"]])

    return df

def print_clusters(df):
    """Prints clusters with their orders."""
    clusters = df.groupby("cluster")
    for cluster, group in clusters:
        print(f"\nCluster {cluster}:")
        for _, row in group.iterrows():
            print(f"  Order {row['order_id']} at ({row['latitude']}, {row['longitude']}) from {row['locality']}")

def plot_clusters(df):
    """Visualizes order clusters using matplotlib."""
    plt.figure(figsize=(8, 6))
    
    unique_clusters = df["cluster"].unique()
    colors = plt.cm.rainbow([i / len(unique_clusters) for i in range(len(unique_clusters))])
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_data = df[df["cluster"] == cluster]
        plt.scatter(cluster_data["longitude"], cluster_data["latitude"], label=f"Cluster {cluster}", color=color)
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Order Clusters")
    plt.legend()
    plt.grid()
    plt.show()
