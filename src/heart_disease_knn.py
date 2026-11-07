import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from evaluate import evaluate
 
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "test_data.csv"
MODEL_LOCATION = ROOT / "model"

df = pd.read_csv(DATASET_PATH, keep_default_na=False)

y = df["Heart_Disease"]
X = df.drop(columns=["Heart_Disease"])

number_cols = X.select_dtypes(include="number").columns.tolist()
text_cols = [c for c in X.columns if c not in number_cols]

preprocessing = ColumnTransformer([
    ("numbers", StandardScaler(), number_cols),
    ("text", OneHotEncoder(handle_unknown="ignore"), text_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def make_pipeline(k):
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", KNeighborsClassifier(n_neighbors=k)),
    ])

print("Trying different values of k (3-fold cross-validation on training data):")
best_k, best_score = None, 0
for k in [3, 5, 11, 21, 51]:
    score = cross_val_score(make_pipeline(k), X_train, y_train,
                            cv=3, scoring="accuracy").mean()
    print(f"  k = {k:<3} accuracy = {score:.4f}")
    if score > best_score:
        best_k, best_score = k, score
print(f"Best k = {best_k}")

model = make_pipeline(best_k)
model.fit(X_train, y_train)

evaluate(model, X_test, y_test, "K-Nearest Neighbours")

joblib.dump(model, MODEL_LOCATION/"heart_knn_model.joblib")
print("\nModel saved as heart_knn_model.joblib")