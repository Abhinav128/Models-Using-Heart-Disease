import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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

model = Pipeline([
    ("preprocess", preprocessing),
    ("model", LogisticRegression(max_iter=1000)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
evaluate(model, X_test, y_test, "Logistic Regression")

joblib.dump(model, MODEL_LOCATION/"heart_logReg_model.joblib")
print("\nModel saved as heart_logReg_model.joblib")