import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from cluster import cluster_orders, print_clusters, plot_clusters  # Import clustering functions

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Sample Data: Orders with descriptions, addresses, and locations
orders = [
    {"order_id": 1, "description": "Buy fresh vegetables and fruits", "address": "Sector 15, Rohini, Delhi", "latitude": 28.7041, "longitude": 77.1025},
    {"order_id": 2, "description": "Order apples and bananas", "address": "Sector 15, Rohini, Delhi", "latitude": 28.7042, "longitude": 77.1026},
    {"order_id": 3, "description": "Milk and dairy products", "address": "Sector 16, Rohini, Delhi", "latitude": 28.7050, "longitude": 77.1035},
    {"order_id": 4, "description": "Electronics - Buy a new smartphone", "address": "Connaught Place, Delhi", "latitude": 28.7005, "longitude": 77.1000},
    {"order_id": 5, "description": "Fresh mangoes and fruits", "address": "Sector 15, Rohini, Delhi", "latitude": 28.7043, "longitude": 77.1027},
]

# Convert to DataFrame
df = pd.DataFrame(orders)

# Perform clustering
df = cluster_orders(df, eps=0.5, min_samples=2)

# Print clustered orders
print_clusters(df)

# Visualize clusters
plot_clusters(df)
