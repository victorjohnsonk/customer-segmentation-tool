
from src.data_generator import generate_customer_data, save_dataset
from src.clustering import run_kmeans
from src.visualization import apply_pca, plot_clusters


def main():
    print("Starting Pipeline...")
    # generate data
    df = generate_customer_data(num_customers=300)
    save_dataset(df)
    # run K-Means clustering
    clustered_df, model = run_kmeans(df, n_clusters=5)
    print("✅ Clustering completed.")
    # apply PCA for visualization
    reduced_df, pca = apply_pca(clustered_df)
    # generate interactive scatter plot
    plot_clusters(reduced_df, save_html="data/customer_clusters.html")
    print("Pipeline complete! Output saved in 'data/customer_clusters.html'")

if __name__ == "__main__":
    main()
