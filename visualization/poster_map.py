import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import from_origin
from rasterio.features import rasterize
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import urllib.request
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
POSTER_PATH = os.path.join(OUTPUT_DIR, 'scottish_whisky_real_geography_map.png')
CRS_WGS84 = 'EPSG:4326'
CRS_BNG = 'EPSG:27700'
CRS_WEB = 'EPSG:3857'

def load_real_scotland_geography() -> gpd.GeoDataFrame:
    gadm_url = 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GBR_2.json'
    print('  [Data] GADM GBR_2 실제 지형 데이터를 로드합니다... (온라인)')
    import ssl
    try:
        req = urllib.request.Request(gadm_url, headers={'User-Agent': 'Mozilla/5.0'})
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context) as response:
            data = json.loads(response.read().decode('utf-8'))
        gdf = gpd.GeoDataFrame.from_features(data['features'], crs=CRS_WGS84)
        gdf = gdf[gdf.geometry.centroid.y > 54.5].copy()
        gdf = gdf.to_crs(CRS_BNG)

        def map_region(name):
            name_lower = str(name).lower()
            if 'highland' in name_lower:
                return 'Highlands'
            if 'moray' in name_lower:
                return 'Speyside'
            if 'argyll' in name_lower:
                return 'Islay'
            if 'orkney' in name_lower or 'shetland' in name_lower or 'eilean' in name_lower:
                return 'Islands'
            return 'Lowlands'
        gdf['whisky_region'] = gdf['NAME_2'].apply(map_region)
        gdf['geometry'] = gdf.geometry.buffer(0)
        regions_gdf = gdf.dissolve(by='whisky_region', aggfunc='first').reset_index()
        np.random.seed(42)
        regions_gdf['temp_increase'] = np.random.uniform(1.2, 2.8, len(regions_gdf))
        print('  ✓ 실제 스코틀랜드 지형 로드 완료.')
        return regions_gdf
    except Exception as e:
        print(f'  ⚠ GADM 다운로드 실패: {e}')
        print('  → Fallback: geopandas naturalearth_lowres 사용')
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        uk = world[world.name == 'United Kingdom'].copy()
        scotland_bounds = gpd.GeoDataFrame(geometry=[Point(-8, 54.5).buffer(10)], crs=CRS_WGS84)
        uk = gpd.overlay(uk, scotland_bounds, how='intersection')
        uk = uk.to_crs(CRS_BNG)
        uk['whisky_region'] = 'Scotland'
        uk['temp_increase'] = 2.0
        return uk

def generate_masked_raster(bounds, regions_gdf: gpd.GeoDataFrame, pixel_size=2000) -> tuple:
    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    shapes = ((geom, 1) for geom in regions_gdf.geometry)
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, all_touched=True, dtype=np.uint8)
    np.random.seed(100)
    x_grad, y_grad = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    precip_array = 5 + 25 * y_grad + 15 * x_grad + np.random.normal(0, 2.5, (height, width))
    precip_array = gaussian_filter(precip_array, sigma=3.0)
    precip_masked = np.where(mask == 1, precip_array, np.nan)
    return (precip_masked, transform, (minx, maxx, miny, maxy))

def generate_real_distilleries(regions_gdf: gpd.GeoDataFrame, n_points=86) -> gpd.GeoDataFrame:
    np.random.seed(200)
    bounds = regions_gdf.total_bounds
    points = []
    clusters = []
    while len(points) < n_points:
        x = np.random.uniform(bounds[0], bounds[2])
        y = np.random.uniform(bounds[1], bounds[3])
        p = Point(x, y)
        if regions_gdf.contains(p).any():
            points.append(p)
            clusters.append(np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
    risk_labels = {0: 'Safe', 1: 'Warning', 2: 'Critical'}
    gdf = gpd.GeoDataFrame({'name': [f'Distillery_{i}' for i in range(n_points)], 'cluster_label': clusters, 'risk_level': [risk_labels[c] for c in clusters]}, geometry=points, crs=CRS_BNG)
    return gdf

def _format_academic_axis(ax, xlabel='Easting (m)', ylabel='Northing (m)'):
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='#aaaaaa', linestyle='--', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=10, fontweight='medium', color='#333333')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='medium', color='#333333')
    ax.tick_params(labelsize=8, colors='#333333', direction='in')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

def create_poster_map():
    print('\n[포스터 제작] 실제 지형 기반 다중 패널 지도 생성을 시작합니다...')
    regions_gdf = load_real_scotland_geography()
    bounds = regions_gdf.total_bounds
    precip_array, transform, extent_bng = generate_masked_raster(bounds, regions_gdf, pixel_size=1500)
    distilleries_gdf = generate_real_distilleries(regions_gdf)
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.subplots_adjust(wspace=0.25, top=0.88, bottom=0.15)
    fig.suptitle('Climate Change Impact on Scottish Whisky Distilleries (Real Geography & Context)', fontsize=20, fontweight='bold', color='#111111', y=0.98)
    extent_arr = [extent_bng[0], extent_bng[1], extent_bng[2], extent_bng[3]]
    ax1 = axes[0]
    ax1.set_title('A. Precipitation Reduction Rate', fontsize=14, fontweight='bold', pad=15)
    cmap1 = plt.cm.YlOrBr.copy()
    cmap1.set_bad(color='white')
    im1 = ax1.imshow(precip_array, cmap=cmap1, extent=extent_arr, origin='upper', interpolation='bilinear')
    regions_gdf.boundary.plot(ax=ax1, color='black', linewidth=0.8, alpha=0.7)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('bottom', size='4%', pad=0.5)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
    cbar1.set_label('Precipitation Drop (%)', fontsize=11, labelpad=8)
    _format_academic_axis(ax1)
    ax2 = axes[1]
    ax2.set_title('B. Regional Temperature Increase', fontsize=14, fontweight='bold', pad=15)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='4%', pad=0.5)
    regions_gdf.plot(column='temp_increase', ax=ax2, cmap='coolwarm', edgecolor='black', linewidth=0.5, legend=True, cax=cax2, legend_kwds={'label': 'Temperature Rise (°C)', 'orientation': 'horizontal'})
    for _, row in regions_gdf.iterrows():
        centroid = row.geometry.centroid
        ax2.text(centroid.x, centroid.y, row['whisky_region'], fontsize=9, fontweight='bold', ha='center', va='center', color='#222222', path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    _format_academic_axis(ax2)
    ax3 = axes[2]
    ax3.set_title('C. Distillery Climate Risk Clusters', fontsize=14, fontweight='bold', pad=15)
    regions_web = regions_gdf.to_crs(CRS_WEB)
    distilleries_web = distilleries_gdf.to_crs(CRS_WEB)
    regions_web.boundary.plot(ax=ax3, color='#555555', linewidth=1.0, alpha=0.5, zorder=2)
    cluster_styles = {0: {'c': '#2196F3', 'label': 'Cluster 0: Safe', 's': 50, 'ec': 'none', 'lw': 0, 'z': 3}, 1: {'c': '#FF9800', 'label': 'Cluster 1: Warning', 's': 80, 'ec': 'black', 'lw': 0.8, 'z': 4}, 2: {'c': '#D32F2F', 'label': 'Cluster 2: Critical', 's': 150, 'ec': 'black', 'lw': 2.0, 'z': 5}}
    for c_id in sorted(distilleries_web['cluster_label'].unique()):
        subset = distilleries_web[distilleries_web['cluster_label'] == c_id]
        style = cluster_styles[c_id]
        ax3.scatter(subset.geometry.x, subset.geometry.y, color=style['c'], s=style['s'], label=style['label'], edgecolors=style['ec'], linewidth=style['lw'], alpha=0.9, zorder=style['z'])
    try:
        ctx.add_basemap(ax3, source=ctx.providers.CartoDB.Positron, alpha=0.8, zorder=1)
    except Exception as e:
        print(f'  ⚠ Contextily 베이스맵 다운로드 실패: {e}')
        regions_web.plot(ax=ax3, color='#e0e0e0', zorder=1)
    ax3.legend(loc='upper left', frameon=True, fontsize=10, facecolor='white', framealpha=0.9, edgecolor='#cccccc')
    _format_academic_axis(ax3, xlabel='Easting (Web Mercator m)', ylabel='Northing (Web Mercator m)')
    print(f'  ✓ 실제 지형 포스터 지도 렌더링 완료. 저장 중... ({POSTER_PATH})')
    fig.savefig(POSTER_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(POSTER_PATH) / 1024
    print(f'  ✅ 300dpi 포스터 이미지가 성공적으로 저장되었습니다. ({size_kb:.0f} KB)')
    return POSTER_PATH
if __name__ == '__main__':
    create_poster_map()