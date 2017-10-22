import pandas as pd
import numpy as np
import os


def generate_customer_data(num_customers=200, random_state=42):
    np.random.seed(random_state)
    customer_ids = np.arange(1, num_customers + 1)
    genders = np.random.choice(['Male', 'Female'], size=num_customers)
    ages = np.random.randint(18, 70, size=num_customers)
    annual_incomes = np.random.normal(60, 20, size=num_customers).clip(15, 120)
    spending_scores = np.random.uniform(1, 100, size=num_customers)

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'AnnualIncome': annual_incomes.round(2),
        'SpendingScore': spending_scores.round(2)
    })
    return df


def save_dataset(df, path="data/synthetic_customers.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Synthetic data saved to {path}")

if __name__ == "__main__":
    df = generate_customer_data()
    save_dataset(df)