import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("📊 Cerinta 2 – Analiză exploratorie")

if 'df_filtered' not in st.session_state:
    st.warning("⚠️ Te rog să încarci și să filtrezi datele în Cerința 1.")
    st.stop()

df = st.session_state['df_filtered'].copy()

st.subheader("Dimensiunea datasetului")
col1, col2 = st.columns(2)
with col1:
    st.metric(" Total Rânduri", df.shape[0])
with col2:
    st.metric(" Total coloane", df.shape[1])

st.subheader("Tipuri de date")
st.dataframe(
    df.dtypes.reset_index()
    .rename(columns={"index": "Coloană", 0: "Tip date"})
)

st.subheader("Coloane cu Valori lipsă")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "Valori lipsă": missing,
    "Procent (%)": missing_pct
}).reset_index().rename(columns={"index": "Coloană"})

only_missing_df = missing_df[missing_df["Valori lipsă"] > 0]

st.dataframe(only_missing_df)

st.subheader("Vizualizare valori lipsă")
fig = px.bar(
    x=df.columns.to_list(),
    y=missing,
    title=f'Frecvența Valorilor Lipsa din Dataframe',
    labels={'x': 'Coloana', 'y': 'Valori lipsa'},
)
fig.update_traces(textposition='outside')
fig.update_xaxes(tickangle=45)
st.plotly_chart(fig, use_container_width=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

st.subheader("Statistici descriptive – coloane numerice")
stats = df[numeric_cols].describe()

st.dataframe(stats)
