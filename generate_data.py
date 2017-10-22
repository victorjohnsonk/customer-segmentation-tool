from src.data_generator import generate_customer_data, save_dataset

def main():
    print("Generating synthetic customer data...!")
    df = generate_customer_data(num_customers=300) #this num can be changed to add more
    save_dataset(df)


if __name__ == "__main__":
    main()