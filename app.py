import streamlit as st

st.set_page_config(
    page_title="Sleep Health EDA",
    page_icon="💤",
    layout="wide"
)

st.title("💤 Sleep Health & Lifestyle – EDA Project")

st.markdown("""
### Descriere proiect
Aplicația permite analiza exploratorie a unui set de date privind
calitatea somnului și stilul de viață, folosind Streamlit.

📌 Navighează folosind meniul din stânga pentru:
- **Cerinta 1** – încărcare și filtrare date
- **Cerinta 2** – analiză statistică și valori lipsă
""")