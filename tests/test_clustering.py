import unittest
from src.data_generator import generate_customer_data
from src.clustering import run_kmeans

class TestClustering(unittest.TestCase):

    def test_cluster_column_exists(self):
        df = generate_customer_data(num_customers=50)
        clustered_df, model = run_kmeans(df, n_clusters=3)
        self.assertIn('Cluster', clustered_df.columns)
        self.assertEqual(len(clustered_df), 50)

    def test_cluster_labels_count(self):
        df = generate_customer_data(num_customers=100)
        clustered_df, model = run_kmeans(df, n_clusters=4)
        self.assertTrue(set(clustered_df['Cluster']).issubset(set(range(4))))

if __name__ == "__main__":
    unittest.main()
