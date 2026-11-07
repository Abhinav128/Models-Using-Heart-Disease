import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


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

preprocessing = ColumnTransformer(
    [("text", OneHotEncoder(handle_unknown="ignore"), text_cols)],
    remainder="passthrough",
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



model = Pipeline([
    ("preprocess", preprocessing),
    ("model", DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, random_state=42)),
])
model.fit(X_train, y_train)
evaluate(model, X_test, y_test, "Tree with max_depth=6, min_samples_leaf=50")

joblib.dump(model, MODEL_LOCATION / "heart_tree_model.joblib")
print("Model saved as heart_tree_model.joblib")