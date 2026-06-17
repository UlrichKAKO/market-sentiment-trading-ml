from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def time_split(df, feature_cols, target_col="target", train_size=0.7):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * train_size)
    X_train = df.iloc[:split_idx][feature_cols]
    y_train = df.iloc[:split_idx][target_col]
    X_test = df.iloc[split_idx:][feature_cols]
    y_test = df.iloc[split_idx:][target_col]
    test_dates = df.iloc[split_idx:]["date"]
    test_returns = df.iloc[split_idx:]["return"]
    return X_train, X_test, y_train, y_test, test_dates, test_returns


def get_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }


def evaluate_classifier(model, X_test, y_test):
    pred = model.predict(X_test)
    auc = None
    if hasattr(model, "predict_proba") and y_test.nunique() > 1:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": auc,
    }
