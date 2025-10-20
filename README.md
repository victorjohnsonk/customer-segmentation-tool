# Customer Segmentation Tool

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-FA0F00?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)

This project segments customers using K-Means clustering and PCA (Principal Component Analysis).
It generates synthetic customer data, runs clustering, and shows results as an interactive scatter plot using Plotly.

---

## Features

- Generate sample customer data
- Perform K-Means clustering
- Apply PCA for 2D visualization
- Create interactive Plotly scatter plots
- Includes a Jupyter notebook for exploration

---

## Requirements

Python 3.6 or later

Install dependencies using:
````pip install -r requirements.txt````

---

## How to Run

From the main project folder, run:
````python -m src.main --clusters 5````

### This will:

1. Generate synthetic data
2. Perform clustering
3. Apply PCA
4. Save the plot to: ````data/customer_clusters_5.html````

Open that HTML file in your browser to view the interactive chart.

---

## Jupyter Notebook

To explore the dataset and clustering visually:
````jupyter notebook notebooks/exploratory_analysis.ipynb````

---

## Running Tests

To verify everything works, run:
````python -m unittest discover -s tests````

---

## Project Structure
````
customer-segmentation-tool/
data/
notebooks/
src/
data_generator.py
clustering.py
visualization.py
config.py
main.py
tests/
test_data_generator.py
test_clustering.py
requirements.txt
README.md
````
---

Notes

This is a simple learning project for practicing Python programming, data generation, and K-Means clustering.
It is meant for beginners who want to understand the basics of data analysis and visualization.
