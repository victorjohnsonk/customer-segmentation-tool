
from src.data_generator import generate_customer_data, save_dataset
from src.clustering import run_kmeans
from src.visualization import apply_pca, plot_clusters
import argparse

def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation Tool")
    parser.add_argument("--clusters", type=int, default=5,
                        help="Number of clusters for K-Means (default: 5)")
    args = parser.parse_args()
    print(f"Starting Pipeline with {args.clusters} clusters...")
    # generate data
    df = generate_customer_data(num_customers=300)
    save_dataset(df)
    # gun K-Means clustering
    clustered_df, model = run_kmeans(df, n_clusters=args.clusters)
    print("✅ Clustering completed.")
    # apply PCA for visualization
    reduced_df, pca = apply_pca(clustered_df)
    # generate interactive scatter plot
    html_path = f"data/customer_clusters_{args.clusters}.html"
    plot_clusters(reduced_df, save_html=html_path)
    print(f"Pipeline complete! Output saved in '{html_path}'")