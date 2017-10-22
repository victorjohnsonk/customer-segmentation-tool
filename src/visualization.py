
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(df, features=None, n_components=2, random_state=42):
    if features is None:
        features = ['Gender', 'Age', 'AnnualIncome', 'SpendingScore')
    pca = PCA(n_components=n_components, random_state=random_state)
    components = pca.fit_transform(df[features])
    df_reduced = df.copy()
    df_reduced['PCA1'] = components[:, 0]
    df_reduced['PCA2'] = components[:, 1]
    print("PCA applied successfully!")
    print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
    return df_reduced, pca