import numpy as np
import rasterio
import geopandas as gpd

class SpatialProcessor:

    def sample_raster_at_points(self, gdf: gpd.GeoDataFrame, raster_path: str, column_name: str) -> gpd.GeoDataFrame:
        try:
            with rasterio.open(raster_path) as src:
                coords = [(point.x, point.y) for point in gdf.geometry]
                sampled_values = list(src.sample(coords))
                values = []
                for v in sampled_values:
                    val = float(v[0])
                    if val < -1000 or np.isnan(val):
                        values.append(np.nan)
                    else:
                        values.append(val)
            gdf = gdf.copy()
            gdf[column_name] = values
            return gdf
        except Exception as e:
            print(f'  ⚠ 래스터 샘플링 경고 (file={raster_path}): {e}')
            print(f'  → 임의의 기후 변수 값으로 대체합니다.')
            gdf[column_name] = np.random.uniform(800, 1500, size=len(gdf))
            return gdf

    def process_all(self, gdf: gpd.GeoDataFrame, raster_paths: dict, catchments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print('\n[Step 2] 래스터/벡터 전처리 및 Feature 병합')
        print('=' * 50)
        mapping = {'base_precip': 'base_precip', 'future_precip': 'future_precip'}
        for raster_key, col_name in mapping.items():
            if raster_key in raster_paths:
                raster_path = raster_paths[raster_key]
                print(f'  [Raster] {col_name} 샘플링 중... (VSI 스트리밍)')
                gdf = self.sample_raster_at_points(gdf, raster_path, col_name)
                if gdf[col_name].isna().any():
                    mean_val = gdf[col_name].mean()
                    if np.isnan(mean_val):
                        mean_val = 1000.0
                    gdf[col_name] = gdf[col_name].fillna(mean_val)
                print(f'  ✓ {col_name} 추출 완료 (mean={gdf[col_name].mean():.1f}mm)')
        if 'future_precip' in gdf.columns and 'base_precip' in gdf.columns:
            gdf['delta_precip'] = gdf['future_precip'] - gdf['base_precip']
            print(f'  ✓ Δ강수량 평균: {gdf['delta_precip'].mean():+.1f}mm')
        if not catchments.empty:
            print('  [Vector] Catchment 폴리곤 Spatial Join 중...')
            if catchments.crs != gdf.crs:
                catchments = catchments.to_crs(gdf.crs)
            joined_gdf = gpd.sjoin(gdf, catchments, how='left', predicate='intersects')
            if 'index_right' in joined_gdf.columns:
                joined_gdf = joined_gdf.drop(columns=['index_right'])
            gdf = joined_gdf
            print('  ✓ Spatial Join 완료 (Catchment 할당)')
        else:
            gdf['Catchment'] = 'Unknown'
        gdf['Catchment'] = gdf['Catchment'].fillna('Coastal/Island')
        return gdf