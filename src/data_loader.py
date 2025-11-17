import streamlit as st
from sklearn import datasets

class DataLoader:
    """Clase responsable de la ingesta de datos."""
    
    @staticmethod
    @st.cache_data
    def load_digits_data():
        """Carga el dataset de dígitos y lo cachea."""
        digits = datasets.load_digits()
        return digits.data, digits.target, digits.images