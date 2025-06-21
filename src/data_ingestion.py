import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values for categorical columns with 'unknown'
    categorical_cols = ["speaker", "job_title", "state_info", "party_affiliation", "context"]
    for col in categorical_cols:
        df[col] = df[col].fillna('unknown')
        df[col] = df[col].astype(str)

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    df["label"] = df["label"].astype(str).str.lower().str.strip()

    def simplify_label(label):
        label_map = {
            "pants-fire": 0,
            "false": 0,
            "barely-true": 0,
            "half-true": 0,
            "mostly-true": 1,
            "true": 1
        }
        return label_map.get(label, -1)

    df["label"] = df["label"].apply(simplify_label)
    df = df[df["label"] != -1]

    return df
