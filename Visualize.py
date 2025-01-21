from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import plotly.express as px
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_2D(X, y): 
    # Initialize t-SNE model with 2 components
    model = TSNE(n_components=2, random_state=42)
    
    # Fit t-SNE to the data
    X_embedded = model.fit_transform(X)
    
    # Visualize the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=15, alpha=0.7)
    plt.colorbar(scatter, label='Digit Label')  # Show class labels as colors
    plt.title("t-SNE Visualization of Digits Dataset")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

def visualize_3D():
    from mpl_toolkits.mplot3d import Axes3D
    
    # Initialize t-SNE with 3 components for 3D visualization
    model = TSNE(n_components=3, random_state=42)
    
    # Fit t-SNE to the data
    X_embedded_3d = model.fit_transform(X)
    
    # Visualize in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_embedded_3d[:, 0], X_embedded_3d[:, 1], X_embedded_3d[:, 2], c=y, cmap='viridis', s=20)
    plt.colorbar(scatter, label='Digit Label')
    ax.set_title("3D t-SNE Visualization of Digits Dataset")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")
    plt.show()

def visualize_interactive():

    # Visualize using Plotly
    fig = px.scatter(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1], 
        color=y, 
        labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'},
        title="t-SNE Visualization with Plotly"
    )
    fig.show()