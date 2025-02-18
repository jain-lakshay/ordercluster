import pandas as pd

def load_orders(file_path="data/orders.csv"):
    """Load order data from CSV."""
    df = pd.read_csv(file_path)
    return df
