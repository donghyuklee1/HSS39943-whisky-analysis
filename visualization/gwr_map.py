import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx
import numpy as np
import os

def draw_gwr_map(gdf: gpd.GeoDataFrame, output_path='output/gwr_sensitivity_map.png'):
    print('[포스터 제작] 1x3 패널 다중 지도(Multi-panel Map) 생성을 시작합니다...')
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), facecolor='white')
    gadm_path = 'output/gadm_scotland.geojson'
    try:
        scotland = gpd.read_file(gadm_path)
    except Exception as e:
        print(f'  ⚠ GADM 데이터 로드 실패: {e}')
        scotland = None

    def setup_panel(ax, title):
        ax.set_title(title, fontsize=16, weight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if scotland is not None:
            scotland.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5, alpha=0.3, zorder=2)
        try:
            cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron, zorder=1)
        except:
            pass
    setup_panel(axes[0], 'Panel 1: Temperature Increase (2000 vs 2050)')
    sc1 = axes[0].scatter(gdf.geometry.x, gdf.geometry.y, s=100, c=gdf['temp_increase'], cmap='YlOrRd', edgecolors='black', linewidth=0.8, alpha=0.85, zorder=3)
    cbar1 = plt.colorbar(sc1, ax=axes[0], shrink=0.6, pad=0.03)
    cbar1.set_label('Temp Increase (°C)', fontsize=12)
    setup_panel(axes[1], 'Panel 2: Precipitation Drop % (2000 vs 2050)')
    sc2 = axes[1].scatter(gdf.geometry.x, gdf.geometry.y, s=100, c=gdf['precip_drop'], cmap='Blues', edgecolors='black', linewidth=0.8, alpha=0.85, zorder=3)
    cbar2 = plt.colorbar(sc2, ax=axes[1], shrink=0.6, pad=0.03)
    cbar2.set_label('Precip Drop (%)', fontsize=12)
    setup_panel(axes[2], 'Panel 3: GWR Local Sensitivity\n(Vulnerability to Precip Drop)')
    sens = gdf['precip_sensitivity'].values
    sens_min, sens_max = (sens.min(), sens.max())
    if sens_max > sens_min:
        sizes = 20 + 480 * ((sens - sens_min) / (sens_max - sens_min))
        sizes = sizes ** 1.2
    else:
        sizes = np.full(len(sens), 100)
    sc3 = axes[2].scatter(gdf.geometry.x, gdf.geometry.y, s=sizes, c=sens, cmap='OrRd', edgecolors='black', linewidth=0.8, alpha=0.85, zorder=3)
    cbar3 = plt.colorbar(sc3, ax=axes[2], shrink=0.6, pad=0.03)
    cbar3.set_label('GWR Coefficient', fontsize=12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ 300dpi 1x3 다중 패널 포스터 이미지가 성공적으로 저장되었습니다.')
    print(f'  → 최종 결과물: {output_path}')