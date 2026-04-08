"""
mystreamlit.py — Despliegue del Modelo Final de Predicción de Depósito Bancario
Autores: Matías Djukic | José María Duro Agea

Uso:
    streamlit run mystreamlit.py
"""

import sys
sys.path.insert(0, '.')

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ─── Obligatorio: redefinir la clase para que joblib pueda deserializar el pipeline ───
class PdaysTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mediana_contactados_ = 0

    def fit(self, X, y=None):
        mask = X['pdays'] != -1
        self.mediana_contactados_ = X.loc[mask, 'pdays'].median() if mask.sum() > 0 else 0
        return self

    def transform(self, X):
        X_trans = X.copy()
        X_trans['contactado_previamente'] = (X_trans['pdays'] != -1).astype(int)
        X_trans.loc[X_trans['pdays'] == -1, 'pdays'] = self.mediana_contactados_
        return X_trans


# ─── Carga del modelo ───────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_final.joblib')

modelo = cargar_modelo()


# ─── Interfaz ───────────────────────────────────────────────────────────────────
st.title("Predicción de la Práctica 1 - Depósito Bancario")
st.markdown(
    "Introduce los datos del cliente para predecir si suscribirá un depósito a plazo."
)

st.header("Datos del Cliente")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Edad", min_value=18, max_value=100, value=40)
    job = st.selectbox("Trabajo", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
        'management', 'retired', 'self-employed', 'services',
        'student', 'technician', 'unemployed', 'unknown'
    ])
    marital = st.selectbox("Estado civil", ['married', 'single', 'divorced'])
    education = st.selectbox("Educación", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox("¿Crédito en mora?", ['no', 'yes'])

with col2:
    balance = st.number_input("Balance anual (€)", value=1000)
    housing = st.selectbox("¿Tiene hipoteca?", ['no', 'yes'])
    loan = st.selectbox("¿Tiene préstamo personal?", ['no', 'yes'])
    contact = st.selectbox("Tipo de contacto", ['cellular', 'telephone', 'unknown'])
    day = st.number_input("Día del último contacto", min_value=1, max_value=31, value=15)

with col3:
    month = st.selectbox("Mes del último contacto", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov'
    ])
    campaign = st.number_input("Nº contactos esta campaña", min_value=1, value=1)
    pdays = st.number_input(
        "Días desde último contacto (-1 = nunca contactado)", 
        min_value=-1, value=-1
    )
    previous = st.number_input("Nº contactos campañas anteriores", min_value=0, value=0)
    poutcome = st.selectbox("Resultado campaña anterior", [
        'unknown', 'failure', 'other', 'success'
    ])


# ─── Predicción ─────────────────────────────────────────────────────────────────
if st.button("PREDECIR", use_container_width=True):

    datos_cliente = pd.DataFrame([{
        'age':       age,
        'job':       job,
        'marital':   marital,
        'education': education,
        'default':   default,
        'balance':   balance,
        'housing':   housing,
        'loan':      loan,
        'contact':   contact,
        'day':       day,
        'month':     month,
        'campaign':  campaign,
        'pdays':     pdays,
        'previous':  previous,
        'poutcome':  poutcome
    }])

    probabilidad = modelo.predict_proba(datos_cliente)[0][1]
    prediccion = 1 if probabilidad >= 0.5 else 0

    st.divider()
    st.header("Resultado")

    if prediccion == 1:
        st.success(f"**SÍ suscribirá el depósito** (probabilidad: {probabilidad:.1%})")
    else:
        st.error(f"**NO suscribirá el depósito** (probabilidad de sí: {probabilidad:.1%})")

    st.progress(float(probabilidad))
    st.caption(f"Probabilidad estimada de contratación: {probabilidad:.4f}")