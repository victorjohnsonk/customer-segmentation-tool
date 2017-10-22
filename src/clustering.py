import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


def preprocess_data(df):
    df = df.copy().reset_index(drop=True)
    # Encode Gender (Male=1, Female=0)
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    # Select numeric columns for clustering
    features = ['Gender', 'Age', 'AnnualIncome', 'SpendingScore']
    X = df[features]
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)   
    return X_scaled, df, scaler, label_encoder


def run_kmeans(df, n_clusters=5, random_state=42):
    X_scaled, processed_df, scaler, label_encoder = preprocess_data(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_scaled)
    processed_df['Cluster'] = kmeans.labels_
    return processed_df, kmeans

