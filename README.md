# Order Clustering Application

A web application that clusters delivery orders based on their geographical locations and address localities using machine learning.
## Preview
![Alt text for the image](https://i.ibb.co/SDChfHYp/Screenshot-2025-02-19-at-1-04-42-PM.png)

## Features

- Clusters orders using DBSCAN algorithm based on:
  - Geographical coordinates (latitude/longitude)
  - Address locality analysis using NLP
  - clustering of orders based on there product description
- Interactive web interface
- Visualization of clusters on a map
- Detailed cluster information display

## Requirements

- Python 3.x
- Required packages (install via `pip`):
  - flask
  - pandas
  - numpy
  - spacy
  - scikit-learn
  - matplotlib
  - fuzzywuzzy
  - geopy
  - python-Levenshtein

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download spaCy's English model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the application:
```bash
python src/main.py
```

2. Open your browser and navigate to `http://localhost:3000`

3. Input your orders data in the following format:
```python
[
    {
        "order_id": 1,
        "description": "Order description",
        "address": "Full address",
        "latitude": 28.7041,
        "longitude": 77.1025
    },
    ...
]
```

4. Click "Process Orders" to see the clustering results

## Project Structure

```
├── data/
│   └── orders.csv         # Sample orders data
├── src/
│   ├── templates/         # HTML templates
│   ├── cluster.py         # Clustering logic
│   ├── main.py           # Flask application
│   └── preprocess.py     # Data preprocessing
└── requirements.txt       # Project dependencies
```

## API Description

- `cluster_orders(df, eps=0.3, min_samples=2)`: Clusters orders using DBSCAN
- `extract_locality(address)`: Extracts locality from address using NLP
- `plot_clusters(df)`: Visualizes clusters on a map

## License

MIT License
