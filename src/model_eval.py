from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

def evaluate(name, y_true, y_pred):
    print(f"\n {name} Set")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

def threshold_tuning(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]

    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1, best_thresh = score, t

    print("Best Threshold:", best_thresh)
    print("Best F1 Score:", best_f1)

    final_preds = (y_probs >= best_thresh).astype(int)
    evaluate(f"Test (Threshold={best_thresh})", y_test, final_preds)
