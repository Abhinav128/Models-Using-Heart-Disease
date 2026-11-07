from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)


def evaluate(model, X_test, y_test, name, X_train=None, y_train=None):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC Score": matthews_corrcoef(y_test, y_pred),
    }

    print(f"\n===== {name}: RESULTS ON TEST DATA =====")
    if X_train is not None and y_train is not None:
        train_acc = accuracy_score(y_train, model.predict(X_train))
        print(f"Train accuracy : {train_acc:.4f}   (on data it learned from)")
    for number, (label, value) in enumerate(metrics.items(), start=1):
        print(f"{number}. {label:<10}: {value:.4f}")

    return metrics