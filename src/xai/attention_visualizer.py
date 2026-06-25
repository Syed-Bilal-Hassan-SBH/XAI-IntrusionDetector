"""
Attention visualization for XAI-E-DiD
Plots attention weights as heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AttentionVisualizer:
    """Visualizes attention weights from the attention mechanism"""
    
    def __init__(self, save_dir: str = "results/xai"):
        """
        Initialize attention visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_attention_weights(
        self,
        attention_weights: np.ndarray,
        save_path: str = None,
        title: str = "Attention Weights",
        figsize: tuple = (12, 6)
    ):
        """
        Plot attention weights as a line plot.
        
        Args:
            attention_weights: Attention weights array (time_steps,)
            save_path: Path to save the plot
            title: Plot title
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        time_steps = np.arange(len(attention_weights))
        
        ax.plot(time_steps, attention_weights, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight peak
        peak_idx = np.argmax(attention_weights)
        peak_value = attention_weights[peak_idx]
        ax.scatter([peak_idx], [peak_value], color='red', s=200, zorder=5, label=f'Peak: {peak_value:.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "attention_weights.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Attention weights plot saved to {save_path}")
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        save_path: str = None,
        title: str = "Attention Heatmap",
        figsize: tuple = (12, 8),
        cmap: str = "YlOrRd"
    ):
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights array (time_steps,)
            save_path: Path to save the plot
            title: Plot title
            figsize: Figure size
            cmap: Colormap
        """
        # Reshape for heatmap if 1D
        if attention_weights.ndim == 1:
            attention_weights = attention_weights.reshape(1, -1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            attention_weights,
            cmap=cmap,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "attention_heatmap.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Attention heatmap saved to {save_path}")
    
    def plot_multi_head_attention(
        self,
        attention_weights: np.ndarray,
        save_path: str = None,
        title: str = "Multi-Head Attention",
        figsize: tuple = (15, 10)
    ):
        """
        Plot multi-head attention weights.
        
        Args:
            attention_weights: Attention weights array (n_heads, time_steps, time_steps)
            save_path: Path to save the plot
            title: Plot title
            figsize: Figure size
        """
        n_heads = attention_weights.shape[0]
        
        fig, axes = plt.subplots(1, n_heads, figsize=figsize)
        
        if n_heads == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            sns.heatmap(
                attention_weights[i],
                cmap='viridis',
                ax=ax,
                cbar=True,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(f'Head {i+1}', fontweight='bold')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "multi_head_attention.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Multi-head attention plot saved to {save_path}")
    
    def plot_attention_summary(
        self,
        attention_weights: np.ndarray,
        save_path: str = None,
        title: str = "Attention Summary"
    ):
        """
        Create a comprehensive attention summary with multiple visualizations.
        
        Args:
            attention_weights: Attention weights array
            save_path: Base path for saving (without extension)
            title: Plot title
        """
        if save_path is None:
            save_path = self.save_dir / "attention_summary"
        
        # Line plot
        self.plot_attention_weights(
            attention_weights,
            save_path=f"{save_path}_line.png",
            title=f"{title} - Line Plot"
        )
        
        # Heatmap
        self.plot_attention_heatmap(
            attention_weights,
            save_path=f"{save_path}_heatmap.png",
            title=f"{title} - Heatmap"
        )
        
        # Statistics
        stats = {
            'mean': float(np.mean(attention_weights)),
            'std': float(np.std(attention_weights)),
            'min': float(np.min(attention_weights)),
            'max': float(np.max(attention_weights)),
            'peak_index': int(np.argmax(attention_weights))
        }
        
        print(f"Attention statistics: {stats}")
        
        return stats


if __name__ == "__main__":
    # Test attention visualizer
    visualizer = AttentionVisualizer()
    
    # Create dummy attention weights
    attention_weights = np.array([0.1, 0.15, 0.3, 0.5, 0.4, 0.2, 0.1, 0.05, 0.08, 0.12])
    
    # Test visualizations
    visualizer.plot_attention_weights(attention_weights)
    visualizer.plot_attention_heatmap(attention_weights)
    visualizer.plot_attention_summary(attention_weights)
    
    # Test multi-head attention
    multi_head = np.random.rand(4, 10, 10)
    visualizer.plot_multi_head_attention(multi_head)
    
    print("Attention visualization tests completed")
