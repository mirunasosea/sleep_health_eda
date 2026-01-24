import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.title("🤖 ML — Problem Setup (Target & Features)")
DATA_PATH = Path("data/Sleep_health_and_lifestyle_dataset.csv")

if "df_categorical_processed" in st.session_state:
    df = st.session_state["df_categorical_processed"]
    data_source = "Procesat (Other aplicat)"
elif "df_filtered" in st.session_state:
    df = st.session_state["df_filtered"]
    data_source = "Filtrat (confirmat)"
elif "df" in st.session_state:
    df = st.session_state["df"]
    data_source = "Original"
else:
    df = pd.read_csv(DATA_PATH)
    data_source = "Original"

st.caption(f"Dataset activ: **{data_source}** | shape: {df.shape[0]} rânduri × {df.shape[1]} coloane")


all_cols = df.columns.tolist()

default_target = "Sleep Disorder" if "Sleep Disorder" in all_cols else all_cols[0]

target_col = st.selectbox(
    "Selectează coloana target pentru clasificare. Recomandare: folositi coloana Sleep Disorder",
    all_cols,
    index=all_cols.index(default_target) if default_target in all_cols else 0
)

st.markdown("### Process Target column")


def make_binary_sleep_disorder_target(d: pd.DataFrame, col="Sleep Disorder") -> pd.Series:
    s = d[col].copy()
    s = s.astype("object")
    s = s.where(~s.isna(), other="No Disorder")
    s = s.astype(str).str.strip().str.lower()

    no_vals = {"no disorder", "none", "nan", "", "null"}
    y = np.where(s.isin(no_vals), 0, 1)
    return pd.Series(y, index=d.index, name="Has Sleep Disorder")

target_mode = "N/A"
y_preview = None
problem_type = "Clasificare"

if target_col == "Sleep Disorder":
    target_mode = st.radio(
        "Cum analizam target-ul din Sleep Disorder?",
        ["Binar (are / nu are disorder)", "Multiclass (tip disorder)"],
        index=0
    )

    if target_mode == "Binar (are / nu are disorder)":
        y_preview = make_binary_sleep_disorder_target(df, col="Sleep Disorder")
    else:
        y_preview = df["Sleep Disorder"].fillna("No Disorder").astype(str)

st.markdown("### Distribuție target (preview)")
if y_preview is None:
    y_tmp = df[target_col]
else:
    y_tmp = y_preview

vc = y_tmp.value_counts(dropna=False)
vp = (y_tmp.value_counts(normalize=True, dropna=False) * 100).round(2)

dist_df = pd.DataFrame({"Count": vc, "Percent (%)": vp})
st.dataframe(dist_df, width='stretch')


st.markdown("### Selectare features (X)")

candidate_features = [c for c in all_cols if c != target_col]
feature_mode = st.radio(
    "Mod selectare features",
    ["Select all (toate în afară de target)", "Exclude columns", "Select manual"],
    index=0
)

selected_features = candidate_features.copy()

if feature_mode == "Exclude columns":
    to_exclude = st.multiselect(
        "Alege coloane de exclus",
        candidate_features,
        default=[]
    )
    selected_features = [c for c in candidate_features if c not in to_exclude]

elif feature_mode == "Select manual":
    selected_features = st.multiselect(
        "Alege coloanele folosite ca features",
        candidate_features,
        default=candidate_features
    )

if len(selected_features) == 0:
    st.error("Trebuie să selectezi cel puțin un feature.")
    st.stop()

st.success(f"✅ Features selectate: {len(selected_features)}")

if y_preview is None:
    y = df[target_col]
else:
    y = y_preview

X = df[selected_features].copy()

if st.button("✅ Confirmă Problem Setup"):
    st.session_state["ml_problem_type"] = problem_type
    st.session_state["ml_target_col"] = target_col
    st.session_state["ml_target_mode"] = target_mode 
    st.session_state["ml_features"] = selected_features

    st.session_state["ml_X"] = X
    st.session_state["ml_y"] = y

    st.success("Problem Setup salvat! Poți merge la pagina următoare (Pipeline).")

with st.expander("🔎 Preview X și y (primele 5 rânduri)"):
    st.write("X.head()")
    st.dataframe(X.head(), width='stretch')
    st.write("y.head()")
    st.dataframe(pd.DataFrame({"y": y}).head(), width='stretch')
