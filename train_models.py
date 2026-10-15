"""
train_models.py
Trains the 4 classifiers FROM SCRATCH (see scratch_ml.py - NumPy only, no scikit-learn)
and saves the artifacts used by the Streamlit app.

Protocol
  1. clean data (drop duplicates), stratified 80:20 train/test split
  2. preprocessing fitted on the TRAIN part only
  3. hyper-parameters + decision threshold chosen by 5-fold stratified CV on TRAIN only
  4. final model refit on the full train set, evaluated ONCE on the untouched test set
"""
import os
import time
import joblib
import numpy as np

from scratch_ml import (
    TARGET, load_and_clean, Preprocessor, stratified_split, tune_model, fit_final,
    evaluate, roc_curve,
    LogisticRegressionScratch, DecisionTreeScratch, KNNScratch, SVMScratch,
)

CSV = "online_shoppers_intention.csv"

print("=" * 62)
print("Step 1: Loading E-Commerce dataset")
print("=" * 62)
if not os.path.exists(CSV):
    raise FileNotFoundError(f"'{CSV}' not found in this folder.")
df, info = load_and_clean(CSV)
print(f"Raw rows: {info['rows_raw']} | missing: {info['rows_missing']} | "
      f"duplicates removed: {info['rows_duplicate']} | clean rows: {info['rows_clean']}")

print("\n" + "=" * 62)
print("Step 2: Preprocessing + 80:20 stratified split")
print("=" * 62)
y = df[TARGET].astype(int).to_numpy()
train_idx, test_idx = stratified_split(y, test_size=0.20)
pre = Preprocessor().fit(df.iloc[train_idx])            # statistics from TRAIN only
X_train, X_test = pre.transform(df.iloc[train_idx]), pre.transform(df.iloc[test_idx])
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {X_train.shape} (purchase rate {y_train.mean():.3%}) | "
      f"Test: {X_test.shape} (purchase rate {y_test.mean():.3%})")

print("\n" + "=" * 62)
print("Step 3: Tuning (5-fold CV on train) + training 4 scratch models")
print("=" * 62)
SPECS = {
    "Logistic Regression": dict(
        cls=LogisticRegressionScratch, file="logistic_regression.pkl", platt=False,
        grid={"l2": [1.0, 10.0, 100.0, 300.0, 1000.0]}, fixed={}),
    "Decision Tree": dict(
        cls=DecisionTreeScratch, file="decision_tree_model.pkl", platt=False,
        grid={"criterion": ["gini", "entropy"], "max_depth": [3, 4, 5, 6, 8],
              "min_samples_leaf": [5, 10, 20], "class_weight": [None, "balanced"]},
        fixed={"min_samples_split": 10}),
    "K-Nearest Neighbors (KNN)": dict(
        cls=KNNScratch, file="knn_model.pkl", platt=False,
        grid={"k": [11, 21, 31, 51, 71, 101], "weights": ["uniform", "distance"]}, fixed={}),
    "Support Vector Machine (SVM)": dict(
        cls=SVMScratch, file="svm_model.pkl", platt=True,
        grid={"C": [0.3, 1.0, 3.0], "gamma": [0.05, 0.1], "class_weight": [None, "balanced"]},
        fixed={"kernel": "rbf", "n_components": 500, "n_iter": 300}),
}

results = {}
for name, spec in SPECS.items():
    print(f"\n>> {name}")
    t0 = time.time()
    tuned = tune_model(spec["cls"], spec["grid"], X_train, y_train,
                       fixed=spec["fixed"], platt=spec["platt"], verbose=False)
    model = fit_final(spec["cls"], tuned, X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    m = evaluate(y_test, prob, model.threshold_)
    fpr, tpr = roc_curve(y_test, prob)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
    results[name] = {
        "Accuracy": round(m["Accuracy"] * 100, 2),
        "Precision": round(m["Precision"] * 100, 2),
        "Recall": round(m["Recall"] * 100, 2),
        "F1-Score": round(m["F1-Score"] * 100, 2),
        "Specificity": round(m["Specificity"] * 100, 2),
        "ROC-AUC": round(m["ROC-AUC"] * 100, 2),
        "Confusion_Matrix": m["Confusion_Matrix"],
        "Threshold": round(model.threshold_, 3),
        "Best_Params": {k: ("None" if v is None else v) for k, v in tuned["best_params"].items()},
        "CV_F1": round(float(tuned["table"].iloc[0]["cv_f1"]) * 100, 2),
        "ROC_Curve": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
    }
    joblib.dump(model, spec["file"])
    print(f"   best params : {tuned['best_params']}")
    print(f"   threshold   : {model.threshold_:.3f}   (CV F1 = {results[name]['CV_F1']}%)")
    print(f"   TEST  Acc={results[name]['Accuracy']}% | Prec={results[name]['Precision']}% | "
          f"Rec={results[name]['Recall']}% | F1={results[name]['F1-Score']}% | "
          f"AUC={results[name]['ROC-AUC']}%   [{time.time() - t0:.0f}s]")

print("\n" + "=" * 62)
print("Step 4: Saving artifacts for the Streamlit app")
print("=" * 62)
joblib.dump(pre, "preprocessor.pkl")
joblib.dump(results, "model_metrics.pkl")
joblib.dump(pre.feature_names_, "feature_names.pkl")
joblib.dump({**info, "n_train": len(y_train), "n_test": len(y_test)}, "dataset_info.pkl")
print("Saved: 4 model .pkl files, preprocessor.pkl, model_metrics.pkl, feature_names.pkl, dataset_info.pkl")
print("\nTraining complete.")
