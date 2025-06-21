# 📁 main.py
from src.data_ingestion import load_and_clean_data
from src.data_transformation import transform_data
from src.model_train import split_and_train
from src.model_eval import evaluate, threshold_tuning


def main():
    # Step 1: Load and preprocess data
    df = load_and_clean_data("src/liar_data/csv file/train.csv")

    # Step 2: Feature transformation
    X, y = transform_data(df)

    # Step 3: Train the model
    model, X_train, X_val, X_test, y_train, y_val, y_test = split_and_train(X, y)

    # Step 4: Evaluate
    evaluate("Train", y_train, model.predict(X_train))
    evaluate("Validation", y_val, model.predict(X_val))
    evaluate("Test", y_test, model.predict(X_test))

    # Step 5: Threshold Tuning
    threshold_tuning(model, X_test, y_test)


if __name__ == "__main__":
    main()
