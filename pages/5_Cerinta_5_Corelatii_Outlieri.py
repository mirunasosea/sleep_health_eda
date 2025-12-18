import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


if 'df_categorical_processed' in st.session_state:
    df = st.session_state['df_categorical_processed']
    st.caption(" Utilizam datasetul PROCESAT")
elif 'df_filtered' in st.session_state:
    df = st.session_state['df_filtered']
    st.caption(" Utilizam datasetul FILTRAT")
elif 'df' in st.session_state:
    df = st.session_state['df']
    st.caption(" Utilizam datasetul ORIGINAL")
else:
    st.warning("⚠️ Te rog să încarci datele în Cerința 1.")
    st.stop()
if df is None:
    st.warning("⚠️ Te rog să încarci datele în Cerința 1.")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Datasetul trebuie să conțină cel puțin două coloane numerice.")
    st.stop()

st.subheader("📊 Matricea de corelatie (Pearson)")

corr_matrix = df[numeric_cols].corr(method="pearson")
fig_corr = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Heatmap corelații Pearson"
)
st.plotly_chart(fig_corr, width="stretch")

st.subheader("🔍 Analiză relație între două variabile numerice")
col_x = st.selectbox("Alege prima variabilă (X)", numeric_cols)
col_y = st.selectbox("Alege a doua variabilă (Y)", numeric_cols, index=1)

scatter_df = df[[col_x, col_y]].dropna()
pearson_corr = scatter_df[col_x].corr(scatter_df[col_y], method="pearson")
st.metric(
    label="Coeficient de corelație Pearson",
    value=f"{pearson_corr:.3f}"
)

fig_scatter = px.scatter(
    scatter_df,
    x=col_x,
    y=col_y,
    title=f"Scatter plot: {col_x} vs {col_y}",
    trendline="ols"
)
st.plotly_chart(fig_scatter, width="stretch")

st.subheader("🚨 Detecția outlierilor (metoda IQR)")

def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper), lower, upper

outlier_summary = []
for col in numeric_cols:
    mask, _, _ = iqr_outliers(df[col])
    count = mask.sum()
    percent = (count / df[col].notna().sum()) * 100

    outlier_summary.append({
        "Coloană": col,
        "Număr outlieri": count,
        "Procent (%)": round(percent, 2)
    })

outlier_df = pd.DataFrame(outlier_summary)
st.subheader("Tabel outlieri (IQR)")
st.dataframe(outlier_df, width="stretch")

st.subheader("Vizualizare outlieri pentru o coloană")

selected_outlier_col = st.selectbox(
    "Selectează o coloană numerică",
    numeric_cols
)
_, lower_fence, upper_fence = iqr_outliers(df[selected_outlier_col])
fig_outliers = px.box(
    df,
    y=selected_outlier_col,
    title=f"Outlieri detectați (IQR) – {selected_outlier_col}",
    points="outliers"
)
fig_outliers.add_hline(y=lower_fence, line_dash="dash", line_color="red", annotation_text="Lower Fence")
fig_outliers.add_hline(y=upper_fence, line_dash="dash", line_color="red", annotation_text="Upper Fence")
    

st.plotly_chart(fig_outliers, width="stretch")




