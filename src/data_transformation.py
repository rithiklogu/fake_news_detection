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
    label_encoders = {}
    encoded_data = []

    for col in ['speaker', 'state_info', 'party_affiliation']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        encoded_data.append(df[col].values.reshape(-1, 1))

    encoded_array = np.hstack(encoded_data)

    count_cols = ['barely_true_counts', 'false_counts', 'half_true_counts', 
                  'mostly_true_counts', 'pants_on_fire_counts']
    X_counts = df[count_cols].astype(float).values

    X_final = np.hstack([X_svd, X_counts, encoded_array])
    y = df["label"].values

    return X_final, y, tfidf, svd, label_encoders
