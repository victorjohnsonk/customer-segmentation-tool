import unittest
from src.data_generator import generate_customer_data

class TestDataGenerator(unittest.TestCase):

    def test_generate_shape_and_columns(self):
        df = generate_customer_data(num_customers=100)
        self.assertEqual(len(df), 100)
        expected_cols = {'CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore'}
        self.assertTrue(expected_cols.issubset(df.columns))

    def test_gender_values(self):
        df = generate_customer_data(num_customers=50)
        self.assertTrue(set(df['Gender']).issubset({'Male', 'Female'}))

if __name__ == "__main__":
    unittest.main()
