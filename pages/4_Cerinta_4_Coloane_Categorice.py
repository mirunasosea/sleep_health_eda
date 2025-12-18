import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("📊 Cerința 4 – Analiza coloanelor categorice")

df = st.session_state.get("df_filtered")

if df is not None:
    st.caption(" Utilizam datasetul FILTRAT")
else:
    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ Te rog să încarci datele în Cerința 1.")
        st.stop()
    st.caption(" Utilizam datasetul ORIGINAL")
    
# identificare coloane categorice
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
if not categorical_cols:
    st.error("Datasetul nu conține coloane categorice.")
    st.stop()

# selectare coloana categorica
selected_col = st.selectbox(
    "Selectează o coloană categorică",
    categorical_cols
)
st.markdown("---")


st.subheader("Distribuția inițială")
freq_abs = df[selected_col].value_counts(dropna=False)
freq_pct = freq_abs / freq_abs.sum() * 100

freq_df = pd.DataFrame({
    "Categorie": freq_abs.index.astype(str),
    "Frecvență": freq_abs.values,
    "Procent (%)": freq_pct.round(2).values
})

# Count plot
fig_bar = px.bar(
    freq_df,
    x="Categorie",
    y="Frecvență",
    text="Frecvență",
    title=f"Distribuția valorilor – {selected_col}"
)
fig_bar.update_traces(textposition="outside")

st.plotly_chart(fig_bar, use_container_width=True)

# Pie chart
fig_pie = px.pie(
    freq_df,
    names="Categorie",
    values="Frecvență",
    title=f"Structura procentuală – {selected_col}"
)
st.plotly_chart(fig_pie, use_container_width=True)

# Tabel
st.subheader("Tabel frecvențe")
st.dataframe(freq_df, use_container_width=True)

# TOP-N categorii
st.markdown("---")
st.subheader("Grupare Top-N categorii")

max_categories = df[selected_col].nunique(dropna=False)

top_n = st.number_input(
    "Afișează Top-N categorii (restul → Other)",
    min_value=1,
    max_value=max_categories,
    value=min(5, max_categories),
    step=1
)

if st.button("Aplica TOP-N categorii"):
    value_counts = df[selected_col].value_counts(dropna=False)
    top_categories = value_counts.nlargest(top_n).index.astype(str)

    df_other = df[selected_col].astype(str).copy()
    df_other[~df_other.isin(top_categories)] = "Other"

    freq_abs_o = df_other.value_counts()
    freq_pct_o = freq_abs_o / freq_abs_o.sum() * 100

    freq_df_o = pd.DataFrame({
        "Categorie": freq_abs_o.index,
        "Frecvență": freq_abs_o.values,
        "Procent (%)": freq_pct_o.round(2).values
    })

    st.markdown("### Distribuția cu Top-N + Other")

    # Count plot
    fig_bar_o = px.bar(
        freq_df_o,
        x="Categorie",
        y="Frecvență",
        text="Frecvență",
        title=f"Distribuția Top-{top_n} + Other – {selected_col}"
    )
    fig_bar_o.update_traces(textposition="outside")
    st.plotly_chart(fig_bar_o, use_container_width=True)

    # Pie chart
    fig_pie_o = px.pie(
        freq_df_o,
        names="Categorie",
        values="Frecvență",
        title=f"Structura procentuală Top-{top_n} + Other – {selected_col}"
    )
    st.plotly_chart(fig_pie_o, use_container_width=True)

    # Tabel
    st.subheader("Tabel frecvențe (Top-N + Other)")
    st.dataframe(freq_df_o, use_container_width=True)

    if st.button("✅ Salvează această transformare pentru urmatorul pas"):
        df_saved = df.copy()
        df_saved[selected_col] = df_saved[selected_col].astype(str)
        df_saved.loc[
            ~df_saved[selected_col].isin(top_categories),
            selected_col
        ] = "Other"

        st.session_state["df_categorical_processed"] = df_saved
        st.success("Datasetul cu Other a fost salvat!")
