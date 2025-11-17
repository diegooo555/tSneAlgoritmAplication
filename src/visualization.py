import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

class Visualizer:
    """Clase responsable de la generación de gráficos UI."""

    @staticmethod
    def plot_sample_images(images, labels, n=10):
        """Genera la visualización estática de las imágenes originales."""
        fig, axes = plt.subplots(1, n, figsize=(15, 2))
        for i in range(n):
            axes[i].imshow(images[i], cmap='gray_r')
            axes[i].set_title(f"Dígito: {labels[i]}")
            axes[i].axis('off')
        return fig

    @staticmethod
    def plot_tsne_result(df, perplexity):
        """Genera el gráfico interactivo de Plotly."""
        fig = px.scatter(
            df, 
            x='Componente 1', 
            y='Componente 2', 
            color='Etiqueta',
            title="Proyección t-SNE del Dataset Digits (0-9)",
            color_discrete_sequence=px.colors.qualitative.G10,
            template="plotly_white",
            hover_data={'Etiqueta': True}
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color='DarkSlateGrey')))
        return fig