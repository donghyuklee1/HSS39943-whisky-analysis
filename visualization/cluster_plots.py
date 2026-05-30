import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import K_RANGE, OUTPUT_DIR, CLUSTER_COLORS

class ClusterPlotter:

    def __init__(self, output_dir: str=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams.update({'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#16213e', 'axes.edgecolor': '#e0e0e0', 'axes.labelcolor': '#e0e0e0', 'xtick.color': '#e0e0e0', 'ytick.color': '#e0e0e0', 'text.color': '#e0e0e0', 'grid.color': '#2a2a4a', 'grid.alpha': 0.5, 'font.size': 11})

    def plot_elbow_silhouette(self, inertias: list, silhouette_scores: list, optimal_k: int) -> str:
        try:
            k_list = list(K_RANGE)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(k_list, inertias, 'o-', color='#e74c3c', linewidth=2, markersize=8)
            ax1.axvline(optimal_k, color='#f1c40f', linestyle='--', linewidth=1.5, label=f'Optimal K={optimal_k}')
            ax1.set_xlabel('Number of Clusters (K)')
            ax1.set_ylabel('Inertia (Within-Cluster SSE)')
            ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True)
            colors = ['#f1c40f' if k == optimal_k else '#3498db' for k in k_list]
            ax2.bar(k_list, silhouette_scores, color=colors, edgecolor='#e0e0e0', linewidth=0.5)
            ax2.set_xlabel('Number of Clusters (K)')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
            ax2.grid(True, axis='y')
            fig.suptitle('K-Means Cluster Optimization — Scottish Whisky Distilleries', fontsize=15, fontweight='bold', color='#e0e0e0', y=1.02)
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, 'elbow_silhouette.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f'  ✓ Elbow/Silhouette 차트 저장: {os.path.basename(filepath)}')
            return filepath
        except Exception as e:
            raise RuntimeError(f'Elbow/Silhouette 차트 생성 실패: {e}') from e

    def plot_cluster_comparison(self, gdf, base_features_cols: list, future_features_cols: list, optimal_k: int) -> str:
        try:
            fig = plt.figure(figsize=(14, 6))
            gs = GridSpec(1, 2, figure=fig)
            base_df = gdf[base_features_cols].copy()
            future_df = gdf[future_features_cols].copy()
            base_df = base_df.fillna(base_df.mean())
            future_df = future_df.fillna(future_df.mean())
            all_features = np.vstack([base_df.values, future_df.values])
            pca = PCA(n_components=2)
            pca.fit(all_features)
            base_2d = pca.transform(base_df.values)
            future_2d = pca.transform(future_df.values)
            ax1 = fig.add_subplot(gs[0, 0])
            for c in range(optimal_k):
                mask = gdf['base_cluster'].values == c
                ax1.scatter(base_2d[mask, 0], base_2d[mask, 1], c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], s=60, alpha=0.8, edgecolors='white', linewidth=0.5, label=f'Cluster {c}')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
            ax1.set_title('Base Period (1990–2020)', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True)
            ax2 = fig.add_subplot(gs[0, 1])
            for c in range(optimal_k):
                mask = gdf['future_cluster'].values == c
                ax2.scatter(future_2d[mask, 0], future_2d[mask, 1], c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], s=60, alpha=0.8, edgecolors='white', linewidth=0.5, label=f'Cluster {c}')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
            ax2.set_title('Future Scenario (2050)', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True)
            fig.suptitle('Cluster Comparison — PCA Projection', fontsize=15, fontweight='bold', color='#e0e0e0', y=1.02)
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, 'cluster_comparison.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f'  ✓ 군집 비교 차트 저장: {os.path.basename(filepath)}')
            return filepath
        except Exception as e:
            raise RuntimeError(f'군집 비교 차트 생성 실패: {e}') from e
if __name__ == '__main__':
    print('cluster_plots.py 모듈 — 독립 실행은 main.py를 통해 진행하세요.')