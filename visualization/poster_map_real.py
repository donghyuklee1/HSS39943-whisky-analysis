import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import from_origin
from rasterio.features import rasterize
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import urllib.request
import json
import ssl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
POSTER_PATH = os.path.join(OUTPUT_DIR, 'scottish_whisky_real_pipeline_map.png')
CRS_WGS84 = 'EPSG:4326'
CRS_BNG = 'EPSG:27700'
CRS_WEB = 'EPSG:3857'

def load_real_scotland_geography() -> gpd.GeoDataFrame:
    gadm_url = 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_GBR_2.json'
    print('  [Geo] GADM 실제 지형 데이터 로드 중...')
    try:
        req = urllib.request.Request(gadm_url, headers={'User-Agent': 'Mozilla/5.0'})
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context) as response:
            data = json.loads(response.read().decode('utf-8'))
        gdf = gpd.GeoDataFrame.from_features(data['features'], crs=CRS_WGS84)
        temp_gdf_bng = gdf.to_crs(CRS_BNG)
        gdf = gdf[temp_gdf_bng.geometry.centroid.y > 550000].copy()
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
        return regions_gdf
    except Exception as e:
        print(f'  ⚠ 지형 로드 실패: {e}')
        raise

def interpolate_climate_raster(regions_gdf: gpd.GeoDataFrame, distilleries_gdf: gpd.GeoDataFrame, target_col: str, pixel_size=2000):
    bounds = regions_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    shapes = ((geom, 1) for geom in regions_gdf.geometry)
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, all_touched=True, dtype=np.uint8)
    points = np.array([(geom.x, geom.y) for geom in distilleries_gdf.geometry])
    values = distilleries_gdf[target_col].values
    grid_x, grid_y = np.meshgrid(np.linspace(minx, maxx, width), np.linspace(maxy, miny, height))
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    grid_z_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
    raster_masked = np.where(mask == 1, grid_z, np.nan)
    return (raster_masked, transform, (minx, maxx, miny, maxy))

def calculate_zonal_statistics(regions_gdf: gpd.GeoDataFrame, distilleries_gdf: gpd.GeoDataFrame, target_col: str):
    zonal_means = []
    for idx, row in regions_gdf.iterrows():
        mask = distilleries_gdf.within(row.geometry)
        pts_in_poly = distilleries_gdf[mask]
        if len(pts_in_poly) > 0:
            mean_val = pts_in_poly[target_col].mean()
        else:
            mean_val = distilleries_gdf[target_col].mean()
        zonal_means.append(mean_val)
    regions_gdf[f'{target_col}_mean'] = zonal_means
    return regions_gdf

def _format_academic_axis(ax, xlabel='Easting (m)', ylabel='Northing (m)'):
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='#aaaaaa', linestyle='--', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=10, fontweight='medium', color='#333333')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='medium', color='#333333')
    ax.tick_params(labelsize=8, colors='#333333', direction='in')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

def create_poster_map_real(distilleries_gdf: gpd.GeoDataFrame):
    print('\n[포스터 제작] 100% Real Data 기반 다중 패널 지도 생성을 시작합니다...')
    regions_gdf = load_real_scotland_geography()
    distilleries_gdf['precip_drop_pct'] = -distilleries_gdf['precip_change_pct']
    precip_raster, transform, extent_bng = interpolate_climate_raster(regions_gdf, distilleries_gdf, target_col='precip_drop_pct', pixel_size=2000)
    regions_gdf = calculate_zonal_statistics(regions_gdf, distilleries_gdf, target_col='temp_increase')
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.subplots_adjust(wspace=0.25, top=0.88, bottom=0.15)
    fig.suptitle('Climate Change Impact on Scottish Whisky (Real API Data & Interpolation)', fontsize=20, fontweight='bold', color='#111111', y=0.98)
    extent_arr = [extent_bng[0], extent_bng[1], extent_bng[2], extent_bng[3]]
    ax1 = axes[0]
    ax1.set_title('A. Precipitation Drop (%)\n(Interpolated from Real Data)', fontsize=14, fontweight='bold', pad=15)
    cmap1 = plt.cm.YlOrBr.copy()
    cmap1.set_bad(color='white')
    im1 = ax1.imshow(precip_raster, cmap=cmap1, extent=extent_arr, origin='upper', interpolation='bilinear')
    regions_gdf.boundary.plot(ax=ax1, color='black', linewidth=0.8, alpha=0.7)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('bottom', size='4%', pad=0.5)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
    cbar1.set_label('Precipitation Drop Rate (%)', fontsize=11, labelpad=8)
    _format_academic_axis(ax1)
    ax2 = axes[1]
    ax2.set_title('B. Regional Avg Temperature Increase\n(Zonal Statistics)', fontsize=14, fontweight='bold', pad=15)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='4%', pad=0.5)
    regions_gdf.plot(column='temp_increase_mean', ax=ax2, cmap='coolwarm', edgecolor='black', linewidth=0.5, legend=True, cax=cax2, legend_kwds={'label': 'Avg Temperature Rise (°C)', 'orientation': 'horizontal'})
    for _, row in regions_gdf.iterrows():
        centroid = row.geometry.centroid
        ax2.text(centroid.x, centroid.y, row['whisky_region'], fontsize=9, fontweight='bold', ha='center', va='center', color='#222222', path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    _format_academic_axis(ax2)
    ax3 = axes[2]
    ax3.set_title('C. Distillery Climate Risk Clusters\n(K-Means on Real Vars)', fontsize=14, fontweight='bold', pad=15)
    regions_web = regions_gdf.to_crs(CRS_WEB)
    distilleries_web = distilleries_gdf.to_crs(CRS_WEB)
    regions_web.boundary.plot(ax=ax3, color='#555555', linewidth=1.0, alpha=0.5, zorder=2)
    cluster_styles = {0: {'c': '#2196F3', 'label': 'Cluster 0: Safe', 's': 50, 'ec': 'none', 'lw': 0, 'z': 3}, 1: {'c': '#FF9800', 'label': 'Cluster 1: Warning', 's': 80, 'ec': 'black', 'lw': 0.8, 'z': 4}, 2: {'c': '#D32F2F', 'label': 'Cluster 2: Critical', 's': 150, 'ec': 'black', 'lw': 2.0, 'z': 5}}
    for c_id in sorted(distilleries_web['cluster_label'].unique()):
        subset = distilleries_web[distilleries_web['cluster_label'] == c_id]
        style = cluster_styles.get(c_id, cluster_styles[0])
        ax3.scatter(subset.geometry.x, subset.geometry.y, color=style['c'], s=style['s'], label=style['label'], edgecolors=style['ec'], linewidth=style['lw'], alpha=0.9, zorder=style['z'])
    try:
        ctx.add_basemap(ax3, source=ctx.providers.CartoDB.Positron, alpha=0.8, zorder=1)
    except Exception as e:
        print(f'  ⚠ Contextily 베이스맵 렌더링 실패: {e}')
        regions_web.plot(ax=ax3, color='#e0e0e0', zorder=1)
    ax3.legend(loc='upper left', frameon=True, fontsize=10, facecolor='white', framealpha=0.9, edgecolor='#cccccc')
    _format_academic_axis(ax3, xlabel='Easting (Web Mercator m)', ylabel='Northing (Web Mercator m)')
    print(f'  ✓ 실제 데이터 보간/포스터 렌더링 완료. 저장 중... ({POSTER_PATH})')
    fig.savefig(POSTER_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(POSTER_PATH) / 1024
    print(f'  ✅ 300dpi 100% 실데이터 포스터 이미지가 성공적으로 저장되었습니다. ({size_kb:.0f} KB)')
    return POSTER_PATH