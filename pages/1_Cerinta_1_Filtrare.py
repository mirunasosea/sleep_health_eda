import streamlit as st
import pandas as pd
import numpy as np

st.title("📌 Cerinta 1 – Încărcare și filtrare date")

uploaded_file = st.file_uploader(
    "Încarcă fișier CSV sau Excel",
    type=["csv", "xlsx"]
)

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df
        

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.session_state['df'] = df
        st.success("Fișier încărcat și citit corect ✅")
    except Exception as e:
        st.error(f"Eroare la citirea fișierului: {e}")


if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("Primele 10 rânduri")
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
            (min_val, max_val)
        )

    st.subheader("🏷️ Filtrare coloane categorice")
    categorical_filters = {}

    for col in categorical_cols:
        options = df[col].dropna().unique().tolist()
        categorical_filters[col] = st.multiselect(
            col,
            options,
        )

    df_filtered = df.copy()

    for col, (low, high) in numeric_filters.items():
        df_filtered = df_filtered[
            (df_filtered[col] >= low) &
            (df_filtered[col] <= high)
        ]

    for col, selected_vals in categorical_filters.items():
        if len(selected_vals) > 0: 
            df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]

    st.info(f"Rânduri înainte de filtrare: {df.shape[0]}")
    st.info(f"Rânduri după filtrare: {df_filtered.shape[0]}")

    st.subheader("Dataset filtrat")
    st.dataframe(df_filtered)

    st.session_state['df_filtered'] = df_filtered


