import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector

st.title("🧪 ML — Pipeline (Preprocesare)")

if "ml_X" not in st.session_state or "ml_y" not in st.session_state:
    st.warning("⚠️ Mai întâi completează pagina ML — Problem Setup.")
    st.stop()

X = st.session_state["ml_X"].copy()
y = st.session_state["ml_y"].copy()
problem_type = st.session_state.get("ml_problem_type", "Clasificare")
st.caption(f"X: {X.shape} | y: {len(y)} | Tip problemă: {problem_type}")


def transform_blood_pressure(df: pd.DataFrame, col="Blood Pressure", mode= "keep"):
    if col not in df.columns:
            return df
    
    if mode == "drop":
        return df.drop(columns=[col])
    
    if mode == "split":
        df_t = pd.concat([df, df[col].str.split('/', expand=True)], axis=1).drop(col, axis=1)
        df_t = df_t.rename(columns={0: f'{col.replace(" ", "_")}_1', 1: f'{col.replace(" ", "_")}_2'})

        df_t[f'{col.replace(" ", "_")}_1'] = df_t[f'{col.replace(" ", "_")}_1'].astype(float)
        df_t[f'{col.replace(" ", "_")}_2'] = df_t[f'{col.replace(" ", "_")}_2'].astype(float)

        return df_t
    
    return df

bp_col = "Blood Pressure"
bp_mode = "keep"

if bp_col in X.columns:
    st.subheader("🩺 Blood Pressure — preprocesare valori")
    st.dataframe(X[bp_col].head(), width='stretch')

    bp_mode = st.radio(
        "Cum tratam coloana Blood presure?",
        ["Păstrează (keep)", "Împarte în 2 coloane numerice (split)", "Elimină (drop)"],
        index=1
    )

    bp_mode_map = {
        "Păstrează (keep)": "keep",
        "Împarte în 2 coloane numerice (split)": "split",
        "Elimină (drop)": "drop"
    }
    bp_mode = bp_mode_map[bp_mode]

    X = transform_blood_pressure(X, bp_col, bp_mode)

with st.expander("🔎 Preview după Blood Pressure processing (primele 5 rânduri)"):
    st.dataframe(X.head(), width='stretch')

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

st.markdown("---")
st.subheader("🔢 Secțiunea Numeric")

numeric_impute = st.selectbox(
    "Imputare numerică (mean / median / most_frequent)",
    ["mean", "median", "most_frequent"],
    index=0
)

scaler_choice = st.selectbox(
    "Scalare numerică",
    ["StandardScaler", "MinMaxScaler", "Fără scalare"],
    index=0
)

if scaler_choice == "StandardScaler":
    scaler = StandardScaler()
elif scaler_choice == "MinMaxScaler":
    scaler = MinMaxScaler()
else:
    scaler = "passthrough"

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy=numeric_impute)),
    ("scaler", scaler),
])

st.markdown("---")
st.subheader("🏷️ Secțiunea Categoric")

st.write("Coloane categorice detectate:", categorical_cols if categorical_cols else "—")

default_cat_fill = "N/A"
cat_fill_value = st.text_input(
    "Valoare de imputare pentru categorice (default: N/A)",
    value=default_cat_fill
)

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=cat_fill_value)),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)),
])

transformers = []
if numeric_cols:
    transformers.append(("num", numeric_pipeline,selector(dtype_include=np.number)))
if categorical_cols:
    transformers.append(("cat", categorical_pipeline, selector(dtype_exclude=np.number)))

if not transformers:
    st.error("Nu există coloane numerice sau categorice de preprocesat.")
    st.stop()

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

full_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
])

X_processed = full_pipeline.fit_transform(X)
processed_columns = []
# numeric
processed_columns.extend(numeric_cols)
# categoric
processed_columns.extend(categorical_cols)
X_processed_df = pd.DataFrame(
    X_processed,
    columns=processed_columns
)

st.markdown("---")
st.subheader("✅ Salvare configurație pipeline")

st.dataframe(X_processed_df.head(10), width='stretch')

st.write("Shape înainte:", X.shape)
st.write("Shape după pipeline:", X_processed.shape)

if st.button("✅ Salvează pipeline-ul pentru antrenare"):
    st.session_state["ml_bp_mode"] = bp_mode
    st.session_state["ml_numeric_cols"] = numeric_cols
    st.session_state["ml_categorical_cols"] = categorical_cols

    st.session_state["ml_numeric_impute"] = numeric_impute
    st.session_state["ml_scaler_choice"] = scaler_choice
    st.session_state["ml_cat_fill_value"] = cat_fill_value

    st.session_state["ml_pipeline_preprocess"] = full_pipeline

    st.success("Pipeline-ul a fost salvat! Poți trece la Train/Test Split.")