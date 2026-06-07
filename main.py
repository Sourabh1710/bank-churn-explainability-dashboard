import warnings

warnings.filterwarnings("ignore")
import os

os.environ["PYTHONWARNINGS"] = "ignore"

# Libraries used
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

# For class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)  # To quiet Optuna logs

import shap
import joblib
from config import DATA_PATH, MODELS_DIR, PLOTS_DIR
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

## STEP 1: Exploratory Data Analysis

df = pd.read_csv(DATA_PATH)


# Dropping Columns which carry no predictive signal:
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

logging.info(f" Dataset shape : {df.shape}")
logging.info(f" Churn rate : {df['Exited'].mean():.1%}")
logging.info(f" Missing values: {df.isnull().sum().sum()}")
logging.info("\n First 3 rows:")
logging.info(df.head(3).to_string())

# EDA PLot 1: Churn Rate by key segments
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Bank Customer Churn - EDA Overview", fontsize=16, fontweight="bold", y=1.01
)

# 1(a) : Overall churn distribution
churn_counts = df["Exited"].value_counts()
axes[0, 0].bar(
    ["Stayed (0)", "Churned (1)"],
    churn_counts.values,
    color=["#4CAF50", "#F44336"],
    edgecolor="white",
    linewidth=1.5,
)
for i, v in enumerate(churn_counts.values):
    axes[0, 0].text(i, v + 50, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=11)
axes[0, 0].set_title("Target Distribution (Class Imbalance ~20%)")
axes[0, 0].set_ylabel("Count")

# 1(b) : Churn by Geography
geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
geo_churn.plot(
    kind="bar", ax=axes[0, 1], color=["#E91E63", "#FF9800", "#2196F3"], rot=0
)
axes[0, 1].set_title("Churn Rate by Geography")
axes[0, 1].set_ylabel("Churn Rate")
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# 1(c) : Churn by Gender
gender_churn = df.groupby("Gender")["Exited"].mean()
axes[0, 2].bar(gender_churn.index, gender_churn.values, color=["#9C27B0", "#00BCD4"])
axes[0, 2].set_title("Churn Rate by Gender")
axes[0, 2].set_ylabel("Churn Rate")
axes[0, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# 1(d) : Churn distribution by Age
df[df["Exited"] == 0]["Age"].plot(
    kind="hist", ax=axes[1, 0], alpha=0.6, bins=30, color="#4CAF50", label="Stayed"
)
df[df["Exited"] == 1]["Age"].plot(
    kind="hist", ax=axes[1, 0], alpha=0.6, bins=30, color="#F44336", label="Churned"
)
axes[1, 0].set_title("Churn Distribution by Age")
axes[1, 0].set_xlabel("Age")
axes[1, 0].legend()

# 1(e) : Churn distribution by Balance
df[df["Exited"] == 0]["Balance"].plot(
    kind="hist", ax=axes[1, 1], alpha=0.6, bins=30, color="#4CAF50", label="Stayed"
)
df[df["Exited"] == 1]["Balance"].plot(
    kind="hist", ax=axes[1, 1], alpha=0.6, bins=30, color="#F44336", label="Churned"
)
axes[1, 1].set_title("Churn Distribution by Balance")
axes[1, 1].set_xlabel("Balance")
axes[1, 1].legend()

# 1(f) : Churn Distribution by Number of Products
prod_churn = df.groupby("NumOfProducts")["Exited"].mean()
axes[1, 2].bar(
    prod_churn.index.astype(str),
    prod_churn.values,
    color=["#3F51B5", "#009688", "#FF5722", "#795548"],
)
axes[1, 2].set_title("Churn Rate by Number of Products")
axes[1, 2].set_xlabel("Number of Products")
axes[1, 2].set_ylabel("Churn Rate")
axes[1, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()

# EDA Plot 2 : Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax,
    linewidths=0.5,
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

## STEP 2 : Feature Engineering
# I engineered these 4 domain features:
#  - credit_score_bucket   = ordinal grouping of creditworthiness
#  - tenure_age_ratio      = how long they've been a customer relative to their age (loyalty signal)
#  - balance_salary_ratio  = Are they keeping much of their salary in the bank? (High ratio = deeper realtionship)
#  - products_per_tenure   = Are they cross_sold quickly or slowly?

df_fe = df.copy()

# Credit Score Bucket divides credit into 5 standard bands used in banking
df_fe["credit_score_bucket"] = pd.cut(
    df_fe["CreditScore"],
    bins=[0, 579, 669, 739, 799, 850],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
)

# Tenure to Age Ratio measures banking loyality relative to life stage
df_fe["tenure_age_ratio"] = df_fe["Tenure"] / (df_fe["Age"] + 1e-9)

# Balance to Salary ratio is an engagement signal, are they depositing a lot vs income?
df_fe["balance_salary_ratio"] = df_fe["Balance"] / (df_fe["EstimatedSalary"] + 1e-9)

# Products per year of tenure ~ cross-sell velocity
df_fe["products_per_tenure"] = df_fe["NumOfProducts"] / (df_fe["Tenure"] + 1)

print(f"Final feature count: {df_fe.shape[1] - 1} features + 1 target")

## STEP 3 : Preprocessing Pipeline
TARGET = "Exited"
DROP_COLS = [TARGET]

num_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "tenure_age_ratio",
    "balance_salary_ratio",
    "products_per_tenure",
]
cat_features = ["Geography", "Gender", "credit_score_bucket"]

X = df_fe.drop(columns=DROP_COLS)
y = df_fe[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logging.info(f" Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
logging.info(f" Train Churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")

# Build the Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_features,
        ),
    ],
    remainder="drop",
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

cat_feature_names = (
    preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features).tolist()
)
all_feature_names = num_features + cat_feature_names
logging.info(f" Features after encoding: {len(all_feature_names)}")

# Applying SMOTE on Training Set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)

logging.info(f" Before SMOTE - Stayed: {(y_train==0).sum()} | Churned: {(y_train==1).sum()}")
logging.info(f" After SMOTE - Stayed: {(y_train_smote==0).sum()} | Churned: {(y_train_smote==1).sum()}")

## STEP 4 : Baseline Model Training
# Three Baseline Models, Which I am going to use are Logistic Regression, Random Forest, XGBoost.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss", verbosity=0,),
}

baseline_results = {}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    baseline_results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "auc": auc,
    }
    logging.info(f" {name:25s} -- AUC-ROC: {auc:.4f}")

## STEP 5 : Hyperparameter Tuning
# GridSearchCV for Random Forest, Optuna for XGBoost.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state =42)

# 5(a) : GridSearchCV for Random Forest
logging.info(" Running GridSearchCV for Random Forest ...")

rf_param_grid = {
    "n_estimators": [100,200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"],  # It's an alternative to SMOTE
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0,
)
rf_grid.fit(X_train_proc, y_train)

best_rf = rf_grid.best_estimator_
rf_proba_tuned = best_rf.predict_proba(X_test_proc)[:, 1]
rf_auc_tuned = roc_auc_score(y_test, rf_proba_tuned)

logging.info(f" Best RF params : {rf_grid.best_params_}")
logging.info(f" RF AUC (tuned) : {rf_auc_tuned:.4f} | baseline: {baseline_results['Random Forest']['auc']:.4f}")

# 5(b) : Optuna for XGBoost
logging.info(" Running Optuna for XGBoost ...")

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(
        model, X_train_proc, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

best_xgb_params = study.best_params
best_xgb_params.update(
    {
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
)

best_xgb = xgb.XGBClassifier(**best_xgb_params)
best_xgb.fit(X_train_proc, y_train)

xgb_proba_tuned = best_xgb.predict_proba(X_test_proc)[:,1]
xgb_auc_tuned = roc_auc_score(y_test, xgb_proba_tuned)

logging.info(f"Best XGB AUC : {xgb_auc_tuned:.4f} | baseline: {baseline_results['XGBoost']['auc']:.4f}")

## STEP 6 : Evaluation - Confusion Matrix, ROC, Precision-Recall
logging.info("\n Generating evaluation plots...")

# collecting final tuned models
eval_models = {
    "Logistic Regression": {
        "proba": baseline_results["Logistic Regression"]["y_proba"],
        "pred": baseline_results["Logistic Regression"]["y_pred"],
    },
    "Random Forest (Tuned)": {
        "proba": rf_proba_tuned,
        "pred": (rf_proba_tuned >= 0.5).astype(int),
    },
    "XGBoost (Tuned)": {
        "proba": xgb_proba_tuned,
        "pred": (xgb_proba_tuned >= 0.5).astype(int),
    },
}

colors = {
    "Logistic Regression": "#2196F3",
    "Random Forest (Tuned)": "#4CAF50",
    "XGBoost (Tuned)": "#FF5722",
}

# Plot: ROC Curves + PR Curves
fig, axes = plt.subplots(1,2, figsize=(14,6))
fig.suptitle("Model Comparison - ROC & Precision-Recall Curves", fontsize=14, fontweight="bold")

# ROC Curves
axes[0].plot([0,1], [0,1], "k--", alpha=0.5, label="Random Baseline")
for name, d in eval_models.items():
    fpr, tpr, _ = roc_curve(y_test,d["proba"])
    auc = roc_auc_score(y_test, d["proba"])
    axes[0].plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC={auc:.3f})")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].grid(alpha=0.3)

# Precision-Recall Curves
for name, d in eval_models.items():
    prec,rec, _ = precision_recall_curve(y_test, d["proba"])
    ap = average_precision_score(y_test, d["proba"])
    axes[1].plot(rec, prec, color=colors[name], lw=2, label=f"{name} (AP={ap:.3f})")
axes[1].axhline(
    y=y_test.mean(),
    color="black",
    linestyle="--",
    alpha=0.5,
    label=f"Baseline ({y_test.mean():.2f})",
)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision_Recall Curves")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot - Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle("Confusion Matrices (Threshold = 0.5)", fontsize=13, fontweight="bold")

for ax, (name, d) in zip(axes, eval_models.items()):
    cm= confusion_matrix(y_test, d["pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10)
    fn = cm[1,0]
    fp = cm[0,1]
    ax.set_xlabel(f"Predicted\n(FN={fn} missed churners, FP={fp} false alarms)")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# Print Classification Reports
logging.info("\n Classification Reports:")
for name, d in eval_models.items():
    auc = roc_auc_score(y_test, d["proba"])
    logging.info(f"{name}")
    logging.info(f" AUC-ROC: {auc:.4f}")
    report = classification_report(
        y_test, d["pred"], target_names=["Stayed","Churned"]
    )
    for line in report.split("\n"):
        logging.info("   " + line)
logging.info(" Evaluation plots saved.")

## STEP 7 : SHAP Values (Explainability)
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_proc)


# Plot 1 : Global feature importance (Beeswarm)
shap_df = pd.DataFrame(shap_values, columns=all_feature_names)

# Mean absolute SHAP per feature
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize = (10, 7))
colors_shap = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(mean_abs_shap[:15])))
mean_abs_shap[:15].plot(kind="barh", ax=ax, color=colors_shap[::-1])
ax.set_title(
    "SHAP Feature Importance - Top 15 Drivers of Churn\n(XGBoost, mean |SHAP| across test set)",
    fontweight="bold",
)
ax.set_xlabel("Mean |SHAP Value| (average impact on prediction)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2 : SHAP Summary Plot (Beeswarm style)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test_proc,
    feature_names=all_feature_names,
    max_display=15,
    show=False,
    plot_type="dot",
)
plt.title("SHAP Summary Plot - feature Impact Direction", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

logging.info(" SHAP plots saved.")
logging.info(f"\n Top 5 Churn Drivers:")
for feat, val in mean_abs_shap.head(5).items():
    logging.info(f" {feat:30s} : {val:.4f}")

## STEP 8 : Business Insights
xgb_importance = pd.Series(
    best_xgb.feature_importances_, index=all_feature_names
).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    "Business Insights - Churn Drivers & Segments", fontsize=14, fontweight="bold"
)

xgb_importance.head(10).plot(kind="barh", ax=axes[0], color="#FF5722")
axes[0].set_title("XGBoost Feature Importance (Top 10)")
axes[0].set_xlabel("Importance Score")
axes[0].invert_yaxis()

df_fe["age_bucket"] = pd.cut(
    df_fe["Age"],
    bins=[18,30,40,50,60,100],
    labels=["18-30","31-40","41-50","51-60","60+"],
)
age_churn = df_fe.groupby("age_bucket", observed=True)["Exited"].mean()
age_churn.plot(
    kind="bar", ax=axes[1], color=sns.color_palette("Reds_d", len(age_churn)), rot=0
)
axes[1].set_title("Churn Rate by Age Group")
axes[1].set_ylabel("Churn Rate")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_business_insights.png", dpi=150, bbox_inches="tight")
plt.close()

## STEP 9 : Save Artifacts
joblib.dump(best_xgb, f"{MODELS_DIR}/xgb_model.pkl")
joblib.dump(best_rf, f"{MODELS_DIR}/rf_model.pkl")
joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.pkl")
joblib.dump(all_feature_names, f"{MODELS_DIR}/feature_names.pkl")
joblib.dump(explainer, f"{MODELS_DIR}/shap_explainer.pkl")

# saving a model summary for the app.
model_summary = {
    "xgb_auc": xgb_auc_tuned,
    "rf_auc": rf_auc_tuned,
    "lr_auc": baseline_results["Logistic Regression"]["auc"],
    "train_size": len(y_train),
    "test_size": len(y_test),
    "churn_rate": float(y.mean()),
    "num_features": len(all_feature_names),
    "optuna_best_params": study.best_params,
}
joblib.dump(model_summary, f"{MODELS_DIR}/model_summary.pkl")

logging.info(f" Artifacts saved to '{MODELS_DIR}/'.")
logging.info(" PIPELINE COMPLETE")
