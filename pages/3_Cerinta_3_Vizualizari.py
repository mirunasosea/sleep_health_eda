import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("📈 Cerința 3 – Analiza unei coloane numerice")

# Verificăm dacă datele există
if 'df_filtered' in st.session_state:
    df = st.session_state['df_filtered']
    st.caption("📌 Analiza se face pe dataset FILTRAT")
elif 'df' in st.session_state:
    df = st.session_state['df']
    st.caption("ℹ️ Analiza se face pe dataset ORIGINAL")
else:
    st.warning("⚠️ Te rog să încarci datele în Cerința 1.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("Datasetul nu conține coloane numerice.")
    st.stop()


selected_col = st.selectbox(
    "Selectează o coloană numerică",
    numeric_cols
)

data = df[selected_col].dropna()

mean_val = data.mean()
median_val = data.median()
std_val = data.std()

st.subheader("Statistici descriptive")

col1, col2, col3 = st.columns(3)

col1.metric("Medie", f"{mean_val:.2f}")
col2.metric("Mediană", f"{median_val:.2f}")
col3.metric("Deviație standard", f"{std_val:.2f}")


bins = st.slider(
    "Număr de bins pentru histogramă",
    min_value=10,
    max_value=100,
    value=30
)


st.subheader("Histogramă")

fig_hist = px.histogram(
    data,
    x=selected_col,
    nbins=bins,
    title=f"Distribuția valorilor – {selected_col}",
    labels={selected_col: selected_col}
)

st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Boxplot")

fig_box = px.box(
    data,
    x=selected_col,
    title=f"Boxplot – {selected_col}",
    points="outliers"
)

st.plotly_chart(fig_box, use_container_width=True)

