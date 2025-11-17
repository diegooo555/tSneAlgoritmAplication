import pandas as pd
from sklearn.manifold import TSNE

class TSNEEngine:
    """Clase responsable de la lógica de Machine Learning."""

    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def run_tsne(self, X, y, perplexity, n_iter, learning_rate):
        """Ejecuta el algoritmo t-SNE y devuelve un DataFrame."""
        
        # Manejo del learning rate 'auto' o numérico
        lr = 'auto' if learning_rate == 'auto' else int(learning_rate)

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=perplexity,
            learning_rate=lr,
            random_state=self.random_state,
            init='random'
        )
        
        projections = tsne.fit_transform(X)
        
        # Estructuración de la salida
        df_plot = pd.DataFrame(projections, columns=['Componente 1', 'Componente 2'])
        df_plot['Etiqueta'] = y.astype(str)
        
        return df_plot