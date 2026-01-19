import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import gradio as gr

df = pd.read_csv("exam_csv.csv")
df.head()

print(df.shape)
df.isna().sum()

X = df.drop(columns=["Potability"])
y = df["Potability"]

num_cols = X.select_dtypes(include = ["int64","float64"]).columns
num_trans = Pipeline (
    steps = [
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ("num",num_trans,num_cols),
    ]
)

for i in num_cols:
    q1 = X[i].quantile(0.25)
    q3 = X[i].quantile(0.75)
    iqr = q3-q1
    lower = q1-1.5*iqr
    upper = q3+1.5*iqr
    print(i,((X[i] < lower) | (X[i] > upper)).sum())

for i in num_cols:
    q1 = X[i].quantile(0.25)
    q3 = X[i].quantile(0.75)
    iqr = q3-q1
    lower = q1-1.5*iqr
    upper = q3+1.5*iqr
    print(i,((X[i] < lower) | (X[i] > upper)).sum())
    X[i] = X[i].clip(lower,upper)

df["ph_bin"] = pd.cut(
    df["ph"],
    bins=[0,6.5,8.5,14],
    labels=["Acidic", "Neutral", "Alkaline"]
)
df["Hardness_bin"] = pd.cut(
    df["Hardness"],
    bins=[0,150,300,600],
    labels=["Soft", "Moderate", "Hard"]
)
df["Turbidity_bin"] = pd.cut(
    df["Turbidity"],
    bins=[0,2,4,6,10],
    labels=["Very Low", "Low", "Medium", "High"]
)
df.head()

clf_rf = RandomForestClassifier(n_estimators=300, random_state=42)
clf_gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
clf_xgb = XGBClassifier(n_estimators=300, objective="binary:logistic", eval_metric="logloss", random_state=42)

stacking_clf = StackingClassifier(
    estimators = [
        ("rf", clf_rf),
        ("gb", clf_gb),
        ("xgb", clf_xgb)
    ],
    final_estimator = LogisticRegression()
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = [
    ("Random Forest", clf_rf),
    ("Gradient Boosting", clf_gb),
    ("XGBoost", clf_xgb),
    ("Stacking Classifier", stacking_clf)
]
for name, clf in models:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")

model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf_rf)
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test,y_pred))


cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print("CV Accuracy scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Std CV Accuracy:", cv_scores.std())

param_grid = {
    "n_estimators": [100,150,175,200],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2,3,5],
    "max_features": ["sqrt", "log2"]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)
print("Best Params:", grid_rf.best_params_)
print("Best CV Accuracy:", grid_rf.best_score_)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}
xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False
)
grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

X_train_clean = X_train.dropna()
y_train_clean = y_train.loc[X_train_clean.index]
X_test_clean = X_test.dropna()
y_test_clean = y_test.loc[X_test_clean.index]
gb_model = GradientBoostingClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_clean, y_train_clean)
print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf_rf)
])
y_pred = model.predict(X_test_clean)
acc = accuracy_score(y_test_clean, y_pred)
prec = precision_score(y_test_clean, y_pred)
rec = recall_score(y_test_clean, y_pred)
f1 = f1_score(y_test_clean, y_pred)
cm = confusion_matrix(y_test_clean, y_pred)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test_clean, y_pred))

with open("water_level.pkl", "wb") as f:
    pickle.dump(model, f)
