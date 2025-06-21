import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def split_and_train(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        random_state=42,
        early_stopping_rounds=10
    )

    model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)

    return model, X_train, X_val, X_test, y_train, y_val, y_test

# def tune_model_with_gridsearch(X, y):
#     X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

#     weights = compute_sample_weight(class_weight="balanced", y=y_train)

#     model = XGBClassifier(
#         objective="binary:logistic",
#         eval_metric="logloss",
#         use_label_encoder=False,
#         random_state=42
#     )

#     param_grid = {
#         'max_depth': [4, 6, 8],
#         'learning_rate': [0.05, 0.1, 0.2],
#         'n_estimators': [100, 200],
#         'min_child_weight': [1, 3],
#         'subsample': [0.8],
#         'colsample_bytree': [0.8]
#     }

#     grid = GridSearchCV(model, param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
#     grid.fit(X_train, y_train, sample_weight=weights)

#     best_model = grid.best_estimator_
#     best_model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)],
#                    early_stopping_rounds=10, verbose=False)

#     return best_model, X_train, X_val, X_test, y_train, y_val, y_test
