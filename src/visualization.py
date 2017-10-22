
import pandas as pd
from sklearn.decomposition import PCA
import plotly.offline as pyo
import plotly.graph_objs as go


def apply_pca(df, features=None, n_components=2, random_state=42):
    if features is None:
        features = ['Gender', 'Age', 'AnnualIncome', 'SpendingScore']
    pca = PCA(n_components=n_components, random_state=random_state)
    components = pca.fit_transform(df[features])
    df_reduced = df.copy()
    df_reduced['PCA1'] = components[:, 0]
    df_reduced['PCA2'] = components[:, 1]
    print("PCA applied successfully!")
    print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
    return df_reduced, pca


def plot_clusters(df, title="Customer Segmentation (PCA Projection)", save_html="cluster_plot.html"):
    if not {'PCA1', 'PCA2', 'Cluster'}.issubset(df.columns):
        raise ValueError("DataFrame must contain PCA1, PCA2, and Cluster columns.")
    trace = go.Scatter(
        x=df['PCA1'],
        y=df['PCA2'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['Cluster'],
            colorscale='Viridis',
            showscale=True
        ),text=[
            f"ID: {row.CustomerID}, Age: {row.Age}, Income: {row.AnnualIncome}, Score: {row.SpendingScore}"
            for row in df.itertuples()],hoverinfo='text')
    layout = go.Layout(
        title=title,
        xaxis=dict(title='PCA 1'),
        yaxis=dict(title='PCA 2'),
        hovermode='closest')
    fig = go.Figure(data=[trace], layout=layout)
    pyo.plot(fig, filename=save_html, auto_open=False)
    print(f"scatter plot saved to: {save_html}")