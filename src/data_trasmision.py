# 📁 src/data_transformation.py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def transform_data(df):
    # Combine text fields
    df["text"] = df["statement"].fillna('') + " " + df["context"].fillna('') + " " + df["job_title"].fillna('')

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df["text"])

    # Dimensionality reduction
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)

    # Encode categorical features
    encoded_cols = []
    for col in ['speaker', 'state_info', 'party_affiliation']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoded_cols.append(col)

    count_cols = ['barely_true_counts', 'false_counts', 'half_true_counts',
                  'mostly_true_counts', 'pants_on_fire_counts']

    X_numeric = df[count_cols + encoded_cols].astype(float).values
    X_final = np.hstack([X_svd, X_numeric])
    y = df["label"]

    return X_final, y
