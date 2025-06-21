import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def transform_data(df):
    df["text"] = df["statement"].fillna('') + " " + df["context"].fillna('') + " " + df["job_title"].fillna('')

    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df["text"])
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    encoded_cols = []
    for col in ['speaker', 'state_info', 'party_affiliation']:
        df[col] = LabelEncoder().fit_transform(df[col])
        encoded_cols.append(col)

    count_cols = ['barely_true_counts', 'false_counts', 'half_true_counts', 
                  'mostly_true_counts', 'pants_on_fire_counts']
    X_numeric = df[count_cols + encoded_cols].astype(float).values

    X_final = np.hstack([X_svd, X_numeric])
    y = df["label"].values

    return X_final, y
