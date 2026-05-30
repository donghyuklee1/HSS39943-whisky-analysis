import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import mgwr
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import spreg

def run_gwr_analysis(gdf: gpd.GeoDataFrame) -> tuple:
    print('[GWR] 지리적 가중 회귀(GWR) 모델링 준비 중...')
    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    if 'precip_drop' not in gdf.columns:
        gdf['precip_drop'] = -gdf['precip_change_pct']
    y_raw = gdf[['humidity_drop']].values
    x_raw = gdf[['precip_drop', 'temp_increase']].values
    scaler_y = StandardScaler()
    scaler_x = StandardScaler()
    y = scaler_y.fit_transform(y_raw)
    X = scaler_x.fit_transform(x_raw)
    print('[GWR] Global OLS 모델 적합 중...')
    ols = spreg.OLS(y, X, name_y='humidity_drop', name_x=['precip_drop', 'temp_increase'])
    ols_r2 = ols.r2
    print(f'  -> Global OLS R-squared: {ols_r2:.4f}')
    print('[GWR] 최적 대역폭(Bandwidth) 동적 탐색 중 (AICc 기준)...')
    bw_selector = Sel_BW(coords, y, X, fixed=False)
    bw = bw_selector.search(criterion='AICc')
    print(f'  -> 최적 대역폭(k-nearest neighbors): {bw}')
    print('[GWR] GWR 모델 적합 중...')
    gwr_model = GWR(coords, y, X, bw=bw, fixed=False)
    gwr_results = gwr_model.fit()
    gwr_r2 = gwr_results.R2
    print(f'  -> GWR R-squared: {gwr_r2:.4f}')
    print(f'[GWR] 설명력(R-squared) 향상폭: {ols_r2:.4f} -> {gwr_r2:.4f} (+{gwr_r2 - ols_r2:.4f})')
    gdf['precip_sensitivity'] = gwr_results.params[:, 1]
    print('[GWR] 강수량 감소에 대한 국지적 민감도 계수(precip_sensitivity) 추출 완료.')
    return (gdf, ols_r2, gwr_r2)