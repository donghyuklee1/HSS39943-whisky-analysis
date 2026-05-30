import geopandas as gpd
from shapely.geometry import Point
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.downloader import DataDownloader

def get_distillery_geodataframe() -> gpd.GeoDataFrame:
    try:
        dl = DataDownloader()
        df = dl.get_whisky_csv()
        geometry = [Point(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        gdf = gdf.rename(columns={'Distillery': 'name', 'Latitude': 'latitude', 'Longitude': 'longitude'})
        flavor_cols = ['Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco', 'Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral']
        print(f'  ✓ 실제 증류소 데이터 로드 완료: {len(gdf)}개 증류소, {len(flavor_cols)}가지 맛 프로필')
        return gdf
    except Exception as e:
        raise RuntimeError(f'증류소 GeoDataFrame 생성 실패: {e}') from e
if __name__ == '__main__':
    gdf = get_distillery_geodataframe()
    print(gdf.head())