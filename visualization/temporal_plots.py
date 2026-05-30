import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_sankey_diagram(labels_df: pd.DataFrame, ownership: np.ndarray, k: int=3, output_path: str=None) -> str:
    import plotly.graph_objects as go
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'sankey_cluster_transition.html')
    cols = list(labels_df.columns)
    years = [col.split('_')[1] for col in cols]
    n_periods = len(cols) - 1
    cluster_names = ['Safe', 'Warning', 'Critical']
    node_labels = []
    node_colors = []
    cluster_colors_rgb = ['rgba(33, 150, 243, 0.85)', 'rgba(255, 152, 0, 0.85)', 'rgba(211, 47, 47, 0.85)']
    for year_idx, year in enumerate(years):
        for c in range(k):
            node_labels.append(f'{year}\n{cluster_names[c]}')
            node_colors.append(cluster_colors_rgb[c])
    sources = []
    targets = []
    values = []
    link_colors = []
    color_independent = 'rgba(255, 87, 34, 0.35)'
    color_corporate = 'rgba(63, 81, 181, 0.35)'
    for period_idx in range(n_periods):
        col_from = cols[period_idx]
        col_to = cols[period_idx + 1]
        for src_cluster in range(k):
            for dst_cluster in range(k):
                ind_mask = (labels_df[col_from] == src_cluster) & (labels_df[col_to] == dst_cluster) & (ownership == 'Independent')
                ind_count = ind_mask.sum()
                if ind_count > 0:
                    source_node = period_idx * k + src_cluster
                    target_node = (period_idx + 1) * k + dst_cluster
                    sources.append(source_node)
                    targets.append(target_node)
                    values.append(int(ind_count))
                    link_colors.append(color_independent)
                corp_mask = (labels_df[col_from] == src_cluster) & (labels_df[col_to] == dst_cluster) & (ownership == 'Corporate')
                corp_count = corp_mask.sum()
                if corp_count > 0:
                    source_node = period_idx * k + src_cluster
                    target_node = (period_idx + 1) * k + dst_cluster
                    sources.append(source_node)
                    targets.append(target_node)
                    values.append(int(corp_count))
                    link_colors.append(color_corporate)
    fig = go.Figure(data=[go.Sankey(arrangement='snap', node=dict(pad=18, thickness=28, line=dict(color='#444444', width=1.5), label=node_labels, color=node_colors), link=dict(source=sources, target=targets, value=values, color=link_colors))])
    fig.update_layout(title=dict(text="Distillery Climate Risk Cluster Transitions (2020 → 2050)<br><sub style='color:#666'>Orange links: Independent distilleries | Blue links: Corporate-owned distilleries</sub>", font=dict(size=16), x=0.5, xanchor='center'), font=dict(size=12, family='Arial, sans-serif'), paper_bgcolor='white', width=1100, height=650, margin=dict(l=30, r=30, t=80, b=30))
    fig.write_html(output_path, include_plotlyjs='cdn')
    size_kb = os.path.getsize(output_path) / 1024
    print(f'  ✓ Sankey Diagram 저장: {os.path.basename(output_path)} ({size_kb:.0f} KB)')
    return output_path

def create_ridgeline_plot(risk_df: pd.DataFrame, ownership: np.ndarray=None, output_path: str=None) -> str:
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'ridgeline_risk_distribution.png')
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'axes.edgecolor': '#333333', 'text.color': '#111111', 'font.family': 'sans-serif'})
    cols = list(risk_df.columns)
    years = [int(col.split('_')[1]) for col in cols]
    n_years = len(years)
    fig, ax = plt.subplots(figsize=(10, 7))
    x_min = risk_df.min().min() - 0.1
    x_max = risk_df.max().max() + 0.2
    x_grid = np.linspace(x_min, x_max, 300)
    cmap = plt.cm.RdYlBu_r
    colors = [cmap(i / (n_years - 1)) for i in range(n_years)]
    overlap = 0.6
    y_spacing = 1.0
    for idx, (col, year) in enumerate(zip(cols, years)):
        values = risk_df[col].dropna().values
        if len(values) < 3:
            continue
        if np.std(values) < 1e-05:
            values = values + np.random.normal(0, 0.001, len(values))
        kde = gaussian_kde(values, bw_method=0.25)
        density = kde(x_grid)
        density_scaled = density / density.max() * overlap
        baseline = idx * y_spacing
        ax.fill_between(x_grid, baseline, baseline + density_scaled, alpha=0.65, color=colors[idx], zorder=n_years - idx + 1)
        ax.plot(x_grid, baseline + density_scaled, color=mcolors.to_rgba(colors[idx], alpha=1.0), linewidth=1.5, zorder=n_years - idx + 2)
        ax.axhline(baseline, color='#cccccc', linewidth=0.5, zorder=0)
        ax.text(x_min - 0.05, baseline + 0.15, str(year), fontsize=13, fontweight='bold', ha='right', va='center', color='#333333')
        if ownership is not None:
            for group, ls, lw_adj in [('Independent', '--', 0.0), ('Corporate', ':', 0.0)]:
                mask = ownership == group
                group_values = values[mask[:len(values)]] if len(mask) >= len(values) else values
                if len(group_values) >= 3:
                    if np.std(group_values) < 1e-05:
                        group_values = group_values + np.random.normal(0, 0.001, len(group_values))
                    kde_g = gaussian_kde(group_values, bw_method=0.3)
                    density_g = kde_g(x_grid)
                    density_g_scaled = density_g / density.max() * overlap * 0.8
                    ax.plot(x_grid, baseline + density_g_scaled, linestyle=ls, linewidth=1.0, color='black' if group == 'Independent' else '#555555', alpha=0.5, zorder=n_years - idx + 3)
    ax.set_xlim(x_min - 0.3, x_max)
    ax.set_ylim(-0.3, n_years * y_spacing + 0.3)
    ax.set_xlabel('Climate Risk Index', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_yticks([])
    ax.set_ylabel('Year', fontsize=13, fontweight='bold', labelpad=35)
    ax.set_title('Risk Distribution Shift Over Time (2020 → 2050)\nRightward tail expansion indicates increasing climate vulnerability', fontsize=14, fontweight='bold', pad=18, color='#222222')
    ax.annotate('← Lower Risk      Higher Risk →', xy=(0.5, -0.08), xycoords='axes fraction', fontsize=10, ha='center', color='#666666')
    if ownership is not None:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='black', linestyle='--', linewidth=1.0, label='Independent Distilleries', alpha=0.6), Line2D([0], [0], color='#555555', linestyle=':', linewidth=1.0, label='Corporate Distilleries', alpha=0.6)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9, frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'  ✓ Ridgeline Plot 저장: {os.path.basename(output_path)} ({size_kb:.0f} KB)')
    return output_path

def run_temporal_visualizations(analysis_results: dict) -> dict:
    print('\n' + '╔' + '═' * 70 + '╗')
    print('║  🎨 시계열 시각화 생성 (Temporal Visualizations)                   ║')
    print('╚' + '═' * 70 + '╝')
    labels_df = analysis_results['labels_df']
    risk_df = analysis_results['risk_df']
    ownership = analysis_results['ownership']
    print('\n[Viz 1] Sankey Diagram (Plotly)')
    print('─' * 50)
    sankey_path = create_sankey_diagram(labels_df, ownership)
    print('\n[Viz 2] Ridgeline Plot (Matplotlib)')
    print('─' * 50)
    ridgeline_path = create_ridgeline_plot(risk_df, ownership)
    return {'sankey_path': sankey_path, 'ridgeline_path': ridgeline_path}
if __name__ == '__main__':
    from models.temporal_analysis import run_temporal_analysis
    analysis = run_temporal_analysis()
    paths = run_temporal_visualizations(analysis)
    print(f'\n  ✅ 시각화 완료:')
    for name, path in paths.items():
        print(f'     ├─ {name}: {path}')