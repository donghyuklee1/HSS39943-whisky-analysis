from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import geopandas as gpd

def perform_real_clustering(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    print('[3/3] 수집된 실제 기후 데이터를 바탕으로 K-Means 클러스터링 수행 중...')
    features = ['temp_increase', 'precip_change_pct', 'humidity_drop']
    if len(gdf) < 3:
        gdf['cluster_label'] = 0
        gdf['risk_level'] = 'Safe'
        return gdf
    X = gdf[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_
    real_centers = scaler.inverse_transform(centers)
    risk_scores = []
    for i, c in enumerate(real_centers):
        score = c[0] - c[1] + c[2]
        risk_scores.append((i, score))
    risk_scores.sort(key=lambda x: x[1])
    mapping = {original_c: new_c for new_c, (original_c, _) in enumerate(risk_scores)}
    mapped_clusters = [mapping[c] for c in clusters]
    gdf['cluster_label'] = mapped_clusters
    risk_labels = {0: 'Safe', 1: 'Warning', 2: 'Critical'}
    gdf['risk_level'] = gdf['cluster_label'].map(risk_labels)
    print('  ✓ 군집화 완료. (0: Safe, 1: Warning, 2: Critical)')
    return gdf