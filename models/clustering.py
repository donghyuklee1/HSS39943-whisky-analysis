import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import K_RANGE, KMEANS_MAX_ITER, KMEANS_N_INIT, RANDOM_SEED

class ClimateClusterAnalyzer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.optimal_k = 3
        self.base_centroids_geo = None
        self.future_centroids_geo = None
        self.inertias = []
        self.silhouette_scores = []
        self.FLAVOR_FEATURES = ['Body', 'Sweetness', 'Smoky', 'Medicinal', 'Tobacco', 'Honey', 'Spicy', 'Winey', 'Nutty', 'Malty', 'Fruity', 'Floral']
        self.BASE_FEATURES = self.FLAVOR_FEATURES + ['base_precip']
        self.FUTURE_FEATURES = self.FLAVOR_FEATURES + ['future_precip']

    def _prepare_features(self, gdf: gpd.GeoDataFrame, columns: list) -> np.ndarray:
        df_features = gdf[columns].copy()
        for col in self.FLAVOR_FEATURES:
            if col not in df_features.columns:
                df_features[col] = 2.0
            else:
                df_features[col] = df_features[col].fillna(df_features[col].mean())
        for col in ['base_precip', 'future_precip']:
            if col in columns:
                df_features[col] = df_features[col].fillna(df_features[col].mean())
        return df_features.values

    def find_optimal_k(self, X_scaled: np.ndarray) -> int:
        self.inertias = []
        self.silhouette_scores = []
        print('\n  K-Means 최적 K 탐색 (Elbow + Silhouette):')
        print('  ' + '-' * 45)
        max_k = min(max(K_RANGE), len(X_scaled) - 1)
        valid_k_range = [k for k in K_RANGE if k <= max_k]
        for k in valid_k_range:
            km = KMeans(n_clusters=k, max_iter=KMEANS_MAX_ITER, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED)
            labels = km.fit_predict(X_scaled)
            self.inertias.append(km.inertia_)
            sil = silhouette_score(X_scaled, labels)
            self.silhouette_scores.append(sil)
            print(f'    K={k:2d}  |  Inertia={km.inertia_:8.1f}  |  Silhouette={sil:.4f}')
        sil_k = valid_k_range[np.argmax(self.silhouette_scores)]
        print(f'  → Silhouette 추천 K = {sil_k}')
        self.optimal_k = sil_k
        print(f'  ★ 최적 K = {self.optimal_k} (Silhouette 기준 채택)')
        return self.optimal_k

    def fit(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        print('\n[Step 3] K-Means 군집 분석 (맛 프로필 + 강수량)')
        print('=' * 50)
        X_base = self._prepare_features(gdf, self.BASE_FEATURES)
        X_base_scaled = self.scaler.fit_transform(X_base)
        optimal_k = self.find_optimal_k(X_base_scaled)
        self.model = KMeans(n_clusters=optimal_k, max_iter=KMEANS_MAX_ITER, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED)
        gdf = gdf.copy()
        gdf['base_cluster'] = self.model.fit_predict(X_base_scaled)
        centroids_geo = []
        for c in range(optimal_k):
            mask = gdf['base_cluster'] == c
            if mask.sum() > 0:
                lat = gdf.loc[mask, 'latitude'].mean()
                lon = gdf.loc[mask, 'longitude'].mean()
            else:
                lat, lon = (56.0, -4.0)
            centroids_geo.append([lat, lon])
        self.base_centroids_geo = np.array(centroids_geo)
        print(f'\n  과거 군집 분포:')
        for c in range(optimal_k):
            count = (gdf['base_cluster'] == c).sum()
            lat, lon = self.base_centroids_geo[c]
            print(f'    Cluster {c}: {count}개 증류소 (중심: {lat:.2f}°N, {lon:.2f}°E)')
        return gdf

    def predict_future(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if self.model is None:
            raise RuntimeError('fit()을 먼저 실행하세요.')
        X_future = self._prepare_features(gdf, self.FUTURE_FEATURES)
        X_future_scaled = self.scaler.transform(X_future)
        gdf = gdf.copy()
        gdf['future_cluster'] = self.model.predict(X_future_scaled)
        future_centroids = []
        for c in range(self.optimal_k):
            mask = gdf['future_cluster'] == c
            if mask.sum() > 0:
                centroid = gdf.loc[mask, ['latitude', 'longitude']].mean().values
            else:
                centroid = self.base_centroids_geo[c]
            future_centroids.append(centroid)
        self.future_centroids_geo = np.array(future_centroids)
        print(f'\n  미래 군집 분포 (2050 시나리오):')
        for c in range(self.optimal_k):
            count = (gdf['future_cluster'] == c).sum()
            lat, lon = self.future_centroids_geo[c]
            print(f'    Cluster {c}: {count}개 증류소 (중심: {lat:.2f}°N, {lon:.2f}°E)')
        gdf['cluster_changed'] = gdf['base_cluster'] != gdf['future_cluster']
        n_changed = gdf['cluster_changed'].sum()
        print(f'\n  ⚠ 군집 이동 증류소: {n_changed}개 / {len(gdf)}개 ({n_changed / len(gdf) * 100:.1f}%)')
        return gdf

    def assign_risk_labels(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        risk_scores = {}
        for c in range(self.optimal_k):
            mask = gdf['future_cluster'] == c
            if mask.sum() > 0:
                mean_delta_precip = gdf.loc[mask, 'delta_precip'].mean()
                risk_scores[c] = -mean_delta_precip
            else:
                risk_scores[c] = 0.0
        sorted_clusters = sorted(risk_scores, key=risk_scores.get, reverse=True)
        risk_labels = ['고위험(Critical)', '경계(Warning)', '주의(Caution)', '안정(Stable)', '저위험(Low Risk)']
        cluster_to_risk = {}
        for rank, cluster_id in enumerate(sorted_clusters):
            label = risk_labels[min(rank, len(risk_labels) - 1)]
            cluster_to_risk[cluster_id] = label
        gdf['risk_level'] = gdf['future_cluster'].map(cluster_to_risk)
        print(f'\n  위험도 라벨 매핑 (강수량 감소폭 기준):')
        for c, label in cluster_to_risk.items():
            count = (gdf['future_cluster'] == c).sum()
            print(f'    Cluster {c} → {label} ({count}개)')
        return gdf
if __name__ == '__main__':
    print('clustering.py 모듈 — 독립 실행은 main.py를 통해 진행하세요.')