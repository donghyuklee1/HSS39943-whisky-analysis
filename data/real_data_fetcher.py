import warnings
import time
import concurrent.futures
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
warnings.filterwarnings('ignore')
CRS_WGS84 = 'EPSG:4326'
CRS_BNG = 'EPSG:27700'

def fetch_distilleries_from_wikidata() -> pd.DataFrame:
    print('[1/3] Wikidata에서 스코틀랜드 위스키 증류소 위치 수집 중...')
    query = '\n    SELECT ?item ?itemLabel ?coord WHERE {\n      { ?item wdt:P31 wd:Q10373548. }\n      UNION\n      { ?item wdt:P31 wd:Q1316882. }\n      ?item wdt:P625 ?coord.\n      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }\n    }\n    '
    url = 'https://query.wikidata.org/sparql'
    headers = {'User-Agent': 'WhiskyClimateAnalysis/1.0', 'Accept': 'application/json'}
    for attempt in range(3):
        try:
            response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if len(data['results']['bindings']) == 0:
                raise ValueError('0 items returned')
            break
        except Exception as e:
            print(f'  ⚠ Wikidata 쿼리 실패 (시도 {attempt + 1}/3): {e}')
            if attempt == 2:
                raise
            time.sleep(2)
    results = []
    for row in data['results']['bindings']:
        name = row['itemLabel']['value']
        coord_str = row['coord']['value']
        if 'Point' in coord_str:
            lon, lat = coord_str.replace('Point(', '').replace(')', '').split()
            results.append({'name': name, 'lon': float(lon), 'lat': float(lat)})
    df = pd.DataFrame(results).drop_duplicates(subset=['name'])
    df = df[(df['lat'] > 54.5) & (df['lat'] < 61.0) & (df['lon'] > -8.5) & (df['lon'] < -0.5)]
    print(f'  ✓ {len(df)}개의 스코틀랜드 증류소 위치 수집 완료.')
    return df

def fetch_climate_data(lat: float, lon: float) -> dict:
    base_url = 'https://climate-api.open-meteo.com/v1/climate'
    params_base = {'latitude': lat, 'longitude': lon, 'start_date': '2000-01-01', 'end_date': '2000-12-31', 'models': 'MPI_ESM1_2_XR', 'daily': 'temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean'}
    params_future = params_base.copy()
    params_future['start_date'] = '2050-01-01'
    params_future['end_date'] = '2050-12-31'
    headers = {'User-Agent': 'WhiskyClimateAnalysis/1.0'}
    try:
        res_base = requests.get(base_url, params=params_base, headers=headers, timeout=10).json()
        temp_base = sum(res_base['daily']['temperature_2m_mean']) / len(res_base['daily']['temperature_2m_mean'])
        precip_base = sum(res_base['daily']['precipitation_sum'])
        rh_list = [x for x in res_base['daily'].get('relative_humidity_2m_mean', []) if x is not None]
        rh_base = sum(rh_list) / len(rh_list) if rh_list else 80.0
        res_future = requests.get(base_url, params=params_future, headers=headers, timeout=10).json()
        temp_future = sum(res_future['daily']['temperature_2m_mean']) / len(res_future['daily']['temperature_2m_mean'])
        precip_future = sum(res_future['daily']['precipitation_sum'])
        rh_list_f = [x for x in res_future['daily'].get('relative_humidity_2m_mean', []) if x is not None]
        rh_future = sum(rh_list_f) / len(rh_list_f) if rh_list_f else 80.0
        return {'temp_increase': temp_future - temp_base, 'precip_change_pct': (precip_future - precip_base) / (precip_base + 1e-05) * 100.0, 'humidity_drop': rh_base - rh_future}
    except Exception as e:
        error_details = ''
        if 'res_base' in locals() and 'error' in res_base:
            error_details = f' Base Error: {res_base.get('reason')}'
        if 'res_future' in locals() and 'error' in res_future:
            error_details += f' Future Error: {res_future.get('reason')}'
        print(f'  ⚠ 기후 API 에러 (lat={lat}, lon={lon}): {e}{error_details}')
        return None

def get_real_distilleries_with_climate(limit=None) -> gpd.GeoDataFrame:
    df = fetch_distilleries_from_wikidata()
    if limit:
        df = df.head(limit)
    total = len(df)
    print(f'[2/3] Open-Meteo Climate API에서 각 증류소의 2000 vs 2050 기후 변화량 병렬 수집 중... (총 {total}개)')
    climate_results = [None] * total

    def fetch_for_index(i):
        time.sleep(0.5)
        row = df.iloc[i]
        return (i, fetch_climate_data(row['lat'], row['lon']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_for_index, i): i for i in range(total)}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i, data = future.result()
            climate_results[i] = data
            completed += 1
            if completed % 20 == 0:
                print(f'      ... {completed}/{total} 처리 완료.')
    climate_results_clean = [res if res is not None else {'temp_increase': float('nan'), 'precip_change_pct': float('nan'), 'humidity_drop': float('nan')} for res in climate_results]
    climate_df = pd.DataFrame(climate_results_clean)
    final_df = pd.concat([df.reset_index(drop=True), climate_df], axis=1)
    final_df = final_df.dropna(subset=['temp_increase', 'precip_change_pct', 'humidity_drop'])
    if len(final_df) == 0:
        raise ValueError('기후 데이터 수집에 모두 실패했습니다.')
    geometry = [Point(xy) for xy in zip(final_df['lon'], final_df['lat'])]
    gdf = gpd.GeoDataFrame(final_df, geometry=geometry, crs=CRS_WGS84)
    gdf = gdf.to_crs(CRS_BNG)
    print('  ✓ 모든 증류소에 대한 기후 데이터 수집 및 EPSG:27700 투영 완료.')
    return gdf
if __name__ == '__main__':
    gdf = get_real_distilleries_with_climate(limit=5)
    print(gdf.head())