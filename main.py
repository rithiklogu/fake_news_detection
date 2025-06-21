from src.data_ingestion import load_and_clean_data
from src.data_transformation import transform_data
from src.model_train import split_and_train  # or tune_model_with_gridsearch
from src.model_eval import evaluate, threshold_tuning
import joblib
import os

def main():
    print("Loading and cleaning data")
    df = load_and_clean_data(r"C:\Users\rithi\Desktop\GEN_AI\fake_news_detection\src\liar_data\csv file\train.csv")

    print("Transforming data...")
    X, y, tfidf, svd, label_encoders = transform_data(df)

    print("Training model...")
    model, X_train, X_val, X_test, y_train, y_val, y_test = split_and_train(X, y)

    print("Evaluating model...")
    evaluate("Validation", y_val, model.predict(X_val))

    print("Final threshold tuning and test evaluation...")
    threshold_tuning(model, X_test, y_test)

    print("Saving model and transformers...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    joblib.dump(svd, "models/svd_transformer.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    print("Saved in /models")

if __name__ == "__main__":
    main()
