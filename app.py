import streamlit as st
import pandas as pd
from src.data_loader import DataLoader
from src.model_engine import TSNEEngine
from src.visualization import Visualizer

# --- Configuración Global ---
st.set_page_config(page_title="Explorador t-SNE", layout="wide")

def main():
    st.title("🔬 Visualizador Interactivo de t-SNE")
    st.markdown("Arquitectura Modular: Datos -> Modelo -> Visualización")

    # 1. Capa de Datos
    try:
        X, y, images = DataLoader.load_digits_data()
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return

    # 2. Visualización Inicial (Raw Data)
    st.subheader("1. Exploración de los Datos (Input)")
    with st.expander("📸 Ver muestra de las imágenes del dataset", expanded=True):
        st.write("Primeras 10 imágenes del dataset (8x8 píxeles).")
        fig_img = Visualizer.plot_sample_images(images, y)
        st.pyplot(fig_img)

    # 3. Interfaz de Configuración (Sidebar)
    st.sidebar.header("⚙️ Configuración del Algoritmo")
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
    n_iter = st.sidebar.slider("Iteraciones", 250, 2000, 1000, 50)
    learning_rate = st.sidebar.selectbox("Learning Rate", ['auto', 10, 50, 100, 200], index=0)
    run_btn = st.sidebar.button("Ejecutar t-SNE")

    # 4. Ejecución y Resultados
    if run_btn:
        with st.spinner('Procesando lógica t-SNE...'):
            # Instanciar motor
            engine = TSNEEngine()
            # Ejecutar lógica
            df_result = engine.run_tsne(X, y, perplexity, n_iter, learning_rate)
            
            # Renderizar
            st.subheader(f"Resultados con Perplexity: {perplexity}")
            fig_tsne = Visualizer.plot_tsne_result(df_result, perplexity)
            st.plotly_chart(fig_tsne, use_container_width=True)
            st.success("¡Procesamiento completado!")
    else:
        st.info("👈 Ajusta los parámetros y presiona 'Ejecutar t-SNE'")
        st.subheader("Data Aplanada")
        st.dataframe(pd.DataFrame(X).head())

if __name__ == "__main__":
    main()