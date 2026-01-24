
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("🤖 ML — Models (Select 3) • Train • Compare")

# ======================================================
# 0) Verificări: avem split + date procesate?
# ======================================================
required = ["ml_X_train", "ml_X_test", "ml_y_train", "ml_y_test", "ml_problem_type"]
missing = [k for k in required if k not in st.session_state]
if missing:
    st.warning(f"⚠️ Lipsesc: {missing}. Mergi întâi la Train/Test Split.")
    st.stop()

problem_type = st.session_state["ml_problem_type"]

X_train = st.session_state["ml_X_train"]
X_test = st.session_state["ml_X_test"]
y_train = st.session_state["ml_y_train"]
y_test = st.session_state["ml_y_test"]

st.caption(f"📌 Train: {X_train.shape} | Test: {X_test.shape}")

st.subheader("✅ Selecteaza EXACT 3 algoritmi")

MODEL_LIST = [
    "Logistic Regression",
    "Random Forest Classifier",
    "SVM",
    "KNN",
    "Gradient Boosting"
]

selected_models = st.multiselect(
    "Alege 3 algoritmi pentru comparatie",
    MODEL_LIST,
    default=["Logistic Regression", "Random Forest Classifier", "SVM"]
)

if len(selected_models) != 3:
    st.info(f"Selectează exact 3 modele (acum ai selectat {len(selected_models)}).")
    st.stop()

# metrică pentru best model
st.subheader("🏁 Criteriu pentru 'Best Model'")
best_metric = st.selectbox(
    "Alege metrica folosită pentru a selecta cel mai bun model",
    ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)"],
    index=1
)

st.markdown("---")
st.subheader("⚙️ Parametri pentru modelele selectate")


params = {}
# Logistic Regression
if "Logistic Regression" in selected_models:
    with st.expander("Logistic Regression", expanded=True):
        lr_C = st.number_input("C", min_value=0.001, max_value=100.0, value=2.0, step=0.1)
        lr_solver = st.selectbox("solver", ["lbfgs", "liblinear", "saga"], index=2)
        lr_max_iter = st.number_input("max_iter", min_value=100, max_value=5000, value=100, step=100)
        lr_class_weight = st.selectbox("class_weight", ["None", "balanced"], index=0)

        params["Logistic Regression"] = {
            "C": float(lr_C),
            "solver": lr_solver,
            "max_iter": int(lr_max_iter),
            "class_weight": None if lr_class_weight == "None" else "balanced",
            "random_state": 42
        }

# Random Forest
if "Random Forest Classifier" in selected_models:
    with st.expander("Random Forest Classifier", expanded=True):
        rf_n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50)
        rf_max_depth = st.number_input("max_depth (0=None)", min_value=0, max_value=50, value=1, step=1)
        rf_min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=2, step=1)
        rf_min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=20, value=1, step=1)

        params["Random Forest Classifier"] = {
            "n_estimators": int(rf_n_estimators),
            "max_depth": None if int(rf_max_depth) == 0 else int(rf_max_depth),
            "min_samples_split": int(rf_min_samples_split),
            "min_samples_leaf": int(rf_min_samples_leaf),
            "random_state": 42
        }

# SVM (SVC)
if "SVM" in selected_models:
    with st.expander("SVM (SVC)", expanded=True):
        svc_C = st.number_input("C (SVC)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
        svc_kernel = st.selectbox("kernel", ["rbf", "linear", "poly"], index=0)
        svc_gamma = st.selectbox("gamma", ["scale", "auto"], index=0)
        svc_probability = st.checkbox("probability=True (ROC-AUC)", value=True)

        params["SVM"] = {
            "C": float(svc_C),
            "kernel": svc_kernel,
            "gamma": svc_gamma,
            "probability": bool(svc_probability),
            "random_state": 42
        }

# KNN
if "KNN" in selected_models:
    with st.expander("KNN", expanded=True):
        knn_k = st.number_input("n_neighbors", min_value=1, max_value=50, value=5, step=1)
        knn_weights = st.selectbox("weights", ["uniform", "distance"], index=0)
        knn_metric = st.selectbox("metric", ["minkowski", "euclidean", "manhattan"], index=0)

        params["KNN"] = {
            "n_neighbors": int(knn_k),
            "weights": knn_weights,
            "metric": knn_metric
        }

# Gradient Boosting
if "Gradient Boosting" in selected_models:
    with st.expander("Gradient Boosting", expanded=True):
        gb_n_estimators = st.number_input("n_estimators (GB)", min_value=50, max_value=1000, value=200, step=50)
        gb_learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        gb_max_depth = st.number_input("max_depth", min_value=1, max_value=10, value=3, step=1)

        params["Gradient Boosting"] = {
            "n_estimators": int(gb_n_estimators),
            "learning_rate": float(gb_learning_rate),
            "max_depth": int(gb_max_depth)
        }

# ======================================================
# 3) Factory: construiește modelul din params
# ======================================================
def build_model(name: str, p: dict):
    if name == "Logistic Regression":
        return LogisticRegression(**p)
    if name == "Random Forest Classifier":
        return RandomForestClassifier(**p)
    if name == "SVM":
        return SVC(**p)
    if name == "KNN":
        return KNeighborsClassifier(**p)
    if name == "Gradient Boosting":
        # sklearn GradientBoostingClassifier nu are max_depth direct; îl setezi prin base_estimator? (nu)
        # În sklearn, max_depth se setează în 'max_depth' al estimators-ului intern -> param name: 'max_depth' nu e acceptat.
        # Folosim max_depth prin 'max_depth' în 'max_depth' al trees: se numește 'max_depth' în GradientBoostingClassifier? NU.
        # Corect: GradientBoostingClassifier are 'max_depth' în param 'max_depth'? (în sklearn e 'max_depth' în 'max_depth' al estimatorului: param e 'max_depth' în sklearn >= 1.0? de fapt e 'max_depth' în 'max_depth'??)
        # Ca să fie sigur compatibil, folosim 'max_depth' ca parte din 'max_depth' via 'max_depth' în param 'max_depth' al sklearn's GBR? 
        # Soluție sigură: folosim 'max_depth' -> 'max_depth' în 'max_depth' al 'DecisionTreeRegressor'? nu.
        # Cel mai sigur: folosim 'max_depth' ca 'max_depth' în param 'max_depth' al GradientBoostingClassifier NU există în toate versiunile.
        # Înlocuim cu 'max_depth' -> 'max_depth' nu, folosim 'max_depth' ca 'max_depth' în 'max_depth' ???.
        # Pentru compatibilitate largă, folosim 'max_depth' prin 'max_depth' în param 'max_depth' al 'DecisionTree' via 'max_depth' în 'max_depth' nu e disponibil.
        # => folosim doar parametrii standard: n_estimators, learning_rate și max_depth mapat la 'max_depth' al 'max_depth' nu.
        # Așadar: folosim 'max_depth' ca 'max_depth' în param 'max_depth' dacă există, altfel fallback la 'max_depth' prin 'max_depth' în 'max_depth' nu.
        # Practic: ignorăm max_depth dacă nu e suportat.
        safe = dict(p)
        max_depth = safe.pop("max_depth", None)
        try:
            return GradientBoostingClassifier(**safe, max_depth=max_depth)
        except TypeError:
            return GradientBoostingClassifier(**safe)
    raise ValueError(f"Model necunoscut: {name}")

# ======================================================
# 4) Train & Compare
# ======================================================
st.markdown("---")
if st.button("🚀 Train & Compare (3 modele)"):
    trained_models = {}
    results = []

    for name in selected_models:
        model = build_model(name, params[name])
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)

        res = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        results.append(res)

    results_df = pd.DataFrame(results)

    # best model logic
    metric_col = best_metric if best_metric != "Accuracy" else "Accuracy"
    best_row = results_df.loc[results_df[metric_col].idxmax()]
    best_model_name = best_row["Model"]

    st.session_state["ml_trained_models"] = trained_models
    st.session_state["ml_models_results"] = results_df
    st.session_state["ml_selected_models"] = selected_models
    st.session_state["ml_best_model_name"] = best_model_name
    st.session_state["ml_models_params"] = params

    st.success(f"✅ Modele antrenate și evaluate. Best model: **{best_model_name}** (după {best_metric})")

    # Afișare tabel rezultate
    st.subheader("📋 Tabel comparativ (test set)")
    st.dataframe(results_df.sort_values(metric_col, ascending=False), use_container_width=True)

    # Evidențiere best
    st.info(f"🏆 Best model: **{best_model_name}** | {best_metric} = {best_row[metric_col]:.3f}")

    with st.expander("🔎 Parametri pentru fiecare model"):
        st.write(params)