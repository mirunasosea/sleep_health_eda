import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

st.title("🧩 ML — Train/Test Split")

if "ml_X" not in st.session_state or "ml_y" not in st.session_state:
    st.warning("⚠️ Mai întâi completează ML — Problem Setup.")
    st.stop()

if "ml_pipeline_preprocess" not in st.session_state:
    st.warning("⚠️ Mai întâi configurează și salvează ML — Pipeline (Preprocesare).")
    st.stop()

X = st.session_state["ml_X"].copy()
y = st.session_state["ml_y"].copy()
problem_type = st.session_state.get("ml_problem_type", "Clasificare")
preprocess_pipeline = st.session_state["ml_pipeline_preprocess"]

st.caption(f"📌 X: {X.shape} | y: {len(y)} | Tip problemă: {problem_type}")


st.subheader("Train/Test Split")

test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.number_input("Random state", min_value=0, max_value=100, value=63, step=1)

if st.button("✅ Generează split și aplică pipeline pe train/test"):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    X_train = preprocess_pipeline.fit_transform(X_train_raw)
    X_test = preprocess_pipeline.fit_transform(X_test_raw)

    # Salvăm în session_state
    st.session_state["ml_test_size"] = float(test_size)
    st.session_state["ml_random_state"] = int(random_state)

    st.session_state["ml_X_train_raw"] = X_train_raw
    st.session_state["ml_X_test_raw"] = X_test_raw
    st.session_state["ml_y_train"] = y_train
    st.session_state["ml_y_test"] = y_test

    st.session_state["ml_X_train"] = X_train
    st.session_state["ml_X_test"] = X_test

    st.success("Split generat")

if "ml_X_train" in st.session_state and "ml_X_test" in st.session_state:
    st.subheader("📌 Rezumat split")
    st.write("Train samples:", st.session_state["ml_X_train"].shape[0])
    st.write("Test samples:", st.session_state["ml_X_test"].shape[0])

    if problem_type == "Clasificare":
        y_train = st.session_state["ml_y_train"]
        y_test = st.session_state["ml_y_test"]

        st.markdown("### Distribuție target (train vs test)")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Train")
            st.dataframe(
                y_train.value_counts(dropna=False)
                .rename_axis("class").to_frame("count"),
                width='stretch'
            )

        with col2:
            st.write("Test")
            st.dataframe(
                y_test.value_counts(dropna=False)
                .rename_axis("class").to_frame("count"),
                width='stretch'
            )

    st.markdown("### Dimensiuni înainte/după preprocesare")
    st.write("X_train_raw:", st.session_state["ml_X_train_raw"].shape)
    st.write("X_train (processed):", st.session_state["ml_X_train"].shape)
    st.write("X_test_raw:", st.session_state["ml_X_test_raw"].shape)
    st.write("X_test (processed):", st.session_state["ml_X_test"].shape)