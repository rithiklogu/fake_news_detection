import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    categorical_cols = ["speaker", "job_title", "state_info", "party_affiliation", "context"]
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    label_map = {
        "pants-fire": 0, "false": 0, "barely-true": 0, "half-true": 0,
        "mostly-true": 1, "true": 1
    }
    df["label"] = df["label"].astype(str).str.lower().str.strip().map(label_map)
    df = df[df["label"].isin([0, 1])]

    return df
