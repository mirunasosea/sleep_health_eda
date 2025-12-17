import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.title("📌 Cerinta 1 – Încărcare și filtrare date")
DATA_PATH = Path("data/Sleep_health_and_lifestyle_dataset.csv")

source = st.radio(
    "Alege sursa datelor:",
    [
        "📂 Încarcă fișier propriu",
        "🧪 Folosește fișierul de test – Sleep Health"
    ]
)

@st.cache_data
def load_test_data(path):
    return pd.read_csv(path)

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

df = None

if source == "📂 Încarcă fișier propriu":
    uploaded_file = st.file_uploader(
        "Încarcă fișier CSV sau Excel",
        type=["csv", "xlsx"]
    )
    if uploaded_file is not None:
        df = load_data(uploaded_file)
elif source == "🧪 Folosește fișierul de test – Sleep Health":
    if DATA_PATH.exists():
        df = load_test_data(DATA_PATH)
    else:
        st.error("Fișierul de test nu a fost găsit.")

if df is not None:
    st.session_state['df'] = df
    st.success("Dataset încărcat cu succes ✅")


if 'df' in st.session_state:
    df = st.session_state['df']

    st.subheader("Primele 10 randuri")
    st.dataframe(df.head(10))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.subheader("🔢 Filtrare coloane numerice")
    numeric_filters = {}

    for col in numeric_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        numeric_filters[col] = st.slider(
            col,
            min_val,
            max_val,
            (min_val, max_val),
            key=f"num_{col}"
        )

    st.subheader("🏷️ Filtrare coloane categorice")
    categorical_filters = {}

    for col in categorical_cols:
        options = df[col].dropna().unique().tolist()
        categorical_filters[col] = st.multiselect(
            col,
            options,
            default=options,
            key=f"cat_{col}"
        )


    df_preview = df.copy()

    for col, (low, high) in numeric_filters.items():
        df_preview = df_preview[
            (df_preview[col] >= low) &
            (df_preview[col] <= high)
        ]

    for col, selected_vals in categorical_filters.items():
        if selected_vals:
            df_preview = df_preview[df_preview[col].isin(selected_vals)]

    st.info(f"Rânduri înainte de filtrare: {df.shape[0]}")
    st.info(f"Rânduri după filtrare: {df_preview.shape[0]}")

    st.subheader("Dataset filtrat (preview live)")
    st.dataframe(df_preview)

    if st.button("✅ Folosește acest dataset în restul aplicației"):
        st.session_state['df_filtered'] = df_preview
        st.success("Dataset filtrat salvat pentru restul cerintelor!")
