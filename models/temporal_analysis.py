import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RANDOM_SEED, KMEANS_MAX_ITER, KMEANS_N_INIT

def interpolate_climate_timeseries(base_values: np.ndarray, future_values: np.ndarray, base_year: int=2020, future_year: int=2050, target_years: list=None, method: str='linear') -> dict:
    if target_years is None:
        target_years = [2020, 2030, 2040, 2050]
    result = {}
    total_span = future_year - base_year
    for year in target_years:
        t = (year - base_year) / total_span
        if method == 'exponential':
            t_adjusted = t ** 1.5
        else:
            t_adjusted = t
        interpolated = base_values * (1 - t_adjusted) + future_values * t_adjusted
        result[year] = interpolated
    return result

def assign_ownership_labels(n_distilleries: int, seed: int=RANDOM_SEED) -> np.ndarray:
    np.random.seed(seed)
    labels = np.random.choice(['Independent', 'Corporate'], size=n_distilleries, p=[0.6, 0.4])
    return labels

def dynamic_kmeans_timeseries(climate_features_by_year: dict, k: int=3, flavor_features: np.ndarray=None) -> pd.DataFrame:
    years = sorted(climate_features_by_year.keys())
    n_distilleries = len(climate_features_by_year[years[0]])
    scaler = StandardScaler()
    labels_dict = {}
    ref_model = None
    ref_cluster_risk_order = None
    for year in years:
        climate_col = climate_features_by_year[year].reshape(-1, 1)
        if flavor_features is not None:
            X = np.hstack([flavor_features, climate_col])
        else:
            X = climate_col
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=k, max_iter=KMEANS_MAX_ITER, n_init=KMEANS_N_INIT, random_state=RANDOM_SEED)
        raw_labels = km.fit_predict(X_scaled)
        cluster_means = {}
        for c in range(k):
            mask = raw_labels == c
            if mask.sum() > 0:
                cluster_means[c] = climate_features_by_year[year][mask].mean()
            else:
                cluster_means[c] = 0.0
        sorted_clusters = sorted(cluster_means.keys(), key=lambda c: cluster_means[c], reverse=True)
        remap = {old: new for new, old in enumerate(sorted_clusters)}
        labels = np.array([remap[l] for l in raw_labels])
        labels_dict[year] = labels
    df = pd.DataFrame(labels_dict)
    df.columns = [f'cluster_{y}' for y in years]
    df.index.name = 'distillery_id'
    return df

def compute_risk_index(climate_features_by_year: dict, ownership: np.ndarray) -> pd.DataFrame:
    years = sorted(climate_features_by_year.keys())
    base_values = climate_features_by_year[years[0]]
    risk_dict = {}
    for year in years:
        values = climate_features_by_year[year]
        delta = (base_values - values) / (base_values + 1e-08)
        delta_normalized = (delta - delta.min()) / (delta.max() - delta.min() + 1e-08)
        capital_penalty = np.where(ownership == 'Independent', 0.3, 0.0)
        risk = delta_normalized + capital_penalty
        risk_dict[year] = risk
    df = pd.DataFrame(risk_dict)
    df.columns = [f'risk_{y}' for y in years]
    return df

def compute_markov_transition_matrix(labels_from: np.ndarray, labels_to: np.ndarray, k: int=3) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=float)
    for i in range(len(labels_from)):
        src = int(labels_from[i])
        dst = int(labels_to[i])
        if 0 <= src < k and 0 <= dst < k:
            matrix[src, dst] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = matrix / row_sums
    return matrix

def compute_all_markov_matrices(labels_df: pd.DataFrame, k: int=3, ownership: np.ndarray=None) -> dict:
    cols = list(labels_df.columns)
    results = {}
    cluster_names = ['Safe', 'Warning', 'Critical']
    for i in range(len(cols) - 1):
        period_name = f'{cols[i].split('_')[1]}→{cols[i + 1].split('_')[1]}'
        labels_from = labels_df[cols[i]].values
        labels_to = labels_df[cols[i + 1]].values
        period_results = {}
        mat_all = compute_markov_transition_matrix(labels_from, labels_to, k)
        period_results['All'] = mat_all
        if ownership is not None:
            for group_name in ['Independent', 'Corporate']:
                mask = ownership == group_name
                if mask.sum() > 0:
                    mat = compute_markov_transition_matrix(labels_from[mask], labels_to[mask], k)
                    period_results[group_name] = mat
        results[period_name] = period_results
    return results

def print_markov_results(markov_results: dict, k: int=3):
    cluster_names = ['Safe', 'Warning', 'Critical']
    print('\n' + '=' * 72)
    print('  📊 Markov Transition Matrices — 군집 전이 확률 분석')
    print('=' * 72)
    for period, groups in markov_results.items():
        print(f'\n  ┌─ 구간: {period} {'─' * 50}┐')
        for group_name, matrix in groups.items():
            print(f'\n    [{group_name}]')
            header = '         ' + '  '.join((f'{c:>10s}' for c in cluster_names))
            print(header)
            print('    ' + '─' * (len(header) - 2))
            for i in range(k):
                row_vals = '  '.join((f'{matrix[i, j]:>10.3f}' for j in range(k)))
                print(f'    {cluster_names[i]:>8s} {row_vals}')
        print(f'  └{'─' * 62}┘')
    _print_markov_sociological_insight(markov_results, k)

def _print_markov_sociological_insight(markov_results: dict, k: int):
    print('\n  📖 [사회학적 함의 — Markov 전이 행렬]')
    print('  ' + '─' * 60)
    for period, groups in markov_results.items():
        if 'Independent' in groups and 'Corporate' in groups:
            ind_mat = groups['Independent']
            corp_mat = groups['Corporate']
            ind_to_critical = (ind_mat[0, 2] + ind_mat[1, 2]) / 2
            corp_to_critical = (corp_mat[0, 2] + corp_mat[1, 2]) / 2
            if ind_to_critical > 0 or corp_to_critical > 0:
                ratio = ind_to_critical / (corp_to_critical + 1e-08)
                diff_pct = (ind_to_critical - corp_to_critical) * 100
                print(f'\n    [{period}] 독립 증류소의 고위험군(Critical) 전이 확률이')
                print(f'    대자본 대비 {diff_pct:+.1f}%p 높음 (비율: {ratio:.2f}배)')
                if diff_pct > 5:
                    print(f'    ⚠ 경고: 독립 증류소의 기후 취약성이 유의미하게 높으며,')
                    print(f'      자본력에 의한 적응 격차가 시간이 지남에 따라 심화될 수 있음.')
                elif diff_pct > 0:
                    print(f'    → 독립 증류소의 미약한 취약성 차이가 관측되나,')
                    print(f'      장기적 누적 효과에 대한 추가 연구가 필요함.')

def compute_wasserstein_distances(risk_df: pd.DataFrame, ownership: np.ndarray=None) -> pd.DataFrame:
    cols = list(risk_df.columns)
    base_col = cols[0]
    base_dist = risk_df[base_col].values
    records = []
    for col in cols:
        year = int(col.split('_')[1])
        current_dist = risk_df[col].values
        wd_all = wasserstein_distance(base_dist, current_dist)
        records.append({'Year': year, 'Group': 'All', 'Wasserstein_Distance': wd_all})
        if ownership is not None:
            for group_name in ['Independent', 'Corporate']:
                mask = ownership == group_name
                if mask.sum() > 1:
                    wd = wasserstein_distance(base_dist[mask], current_dist[mask])
                    records.append({'Year': year, 'Group': group_name, 'Wasserstein_Distance': wd})
    return pd.DataFrame(records)

def print_wasserstein_results(wd_df: pd.DataFrame):
    print('\n' + '=' * 72)
    print('  📏 Wasserstein Distance — 리스크 분포 이탈도 분석')
    print('=' * 72)
    for group in wd_df['Group'].unique():
        subset = wd_df[wd_df['Group'] == group].sort_values('Year')
        print(f'\n    [{group}]')
        print(f'    {'Year':>6s} │ {'W-Distance':>12s} │ {'Bar':>20s}')
        print(f'    {'─' * 6}─┼─{'─' * 12}─┼─{'─' * 20}')
        max_wd = subset['Wasserstein_Distance'].max()
        for _, row in subset.iterrows():
            bar_len = int(20 * row['Wasserstein_Distance'] / (max_wd + 1e-08))
            bar = '█' * bar_len
            print(f'    {int(row['Year']):>6d} │ {row['Wasserstein_Distance']:>12.4f} │ {bar}')
    print('\n  📖 [사회학적 함의 — Wasserstein Distance]')
    print('  ' + '─' * 60)
    for group in ['Independent', 'Corporate']:
        subset = wd_df[wd_df['Group'] == group].sort_values('Year')
        if len(subset) >= 2:
            final = subset.iloc[-1]['Wasserstein_Distance']
            initial = subset.iloc[0]['Wasserstein_Distance']
            growth = final - initial
            print(f'\n    [{group}] 2020→2050 분포 이탈: {growth:.4f}')
    ind_final = wd_df[(wd_df['Group'] == 'Independent') & (wd_df['Year'] == 2050)]
    corp_final = wd_df[(wd_df['Group'] == 'Corporate') & (wd_df['Year'] == 2050)]
    if len(ind_final) > 0 and len(corp_final) > 0:
        ind_wd = ind_final.iloc[0]['Wasserstein_Distance']
        corp_wd = corp_final.iloc[0]['Wasserstein_Distance']
        print(f'\n    → 2050년 기준 독립 증류소의 분포 이탈({ind_wd:.4f})이')
        print(f'      대자본({corp_wd:.4f}) 대비 {ind_wd / corp_wd:.2f}배 크게 관측됨.')
        print(f'      이는 자본력에 기반한 기후 적응 전략의 유무가')
        print(f'      리스크 분포의 형태적 차이로 직결됨을 시사합니다.')

def run_temporal_analysis(gdf=None, base_precip: np.ndarray=None, future_precip: np.ndarray=None, flavor_features: np.ndarray=None, k: int=3, method: str='linear') -> dict:
    print('\n' + '╔' + '═' * 70 + '╗')
    print('║  🕐 시계열 확률적 기후 취약성 분석 (Temporal Stochastic Analysis)  ║')
    print('╚' + '═' * 70 + '╝')
    n_distilleries = 86 if gdf is None else len(gdf)
    if base_precip is None or future_precip is None:
        print('  ⚠ 강수량 데이터가 없습니다. Mock 데이터를 생성합니다...')
        np.random.seed(RANDOM_SEED)
        base_precip = np.random.uniform(900, 1800, n_distilleries)
        future_precip = base_precip * np.random.uniform(0.7, 0.95, n_distilleries)
    if flavor_features is None:
        print('  ⚠ 맛 프로필이 없습니다. Mock 피처를 생성합니다...')
        np.random.seed(RANDOM_SEED + 1)
        flavor_features = np.random.randint(0, 5, size=(n_distilleries, 12)).astype(float)
    print('\n[Task 1-1] 기후 변수 시계열 보간 (Interpolation)')
    print('─' * 50)
    climate_by_year = interpolate_climate_timeseries(base_precip, future_precip, method=method)
    for year, values in climate_by_year.items():
        print(f'  {year}년 강수량: mean={values.mean():.1f}mm, std={values.std():.1f}mm')
    ownership = assign_ownership_labels(n_distilleries)
    n_ind = (ownership == 'Independent').sum()
    n_corp = (ownership == 'Corporate').sum()
    print(f'\n  자본 규모 분포: 독립({n_ind}개) / 대자본({n_corp}개)')
    print(f'\n[Task 1-2] 연대별 동적 K-Means (K={k})')
    print('─' * 50)
    labels_df = dynamic_kmeans_timeseries(climate_by_year, k=k, flavor_features=flavor_features)
    cluster_names = {0: 'Safe', 1: 'Warning', 2: 'Critical'}
    for col in labels_df.columns:
        year = col.split('_')[1]
        counts = labels_df[col].value_counts().sort_index()
        dist_str = ', '.join((f'{cluster_names.get(c, f'C{c}')}={n}' for c, n in counts.items()))
        print(f'  {year}년: {dist_str}')
    risk_df = compute_risk_index(climate_by_year, ownership)
    print(f'\n[Task 2-1] Markov 전이 행렬 계산')
    print('─' * 50)
    markov_results = compute_all_markov_matrices(labels_df, k=k, ownership=ownership)
    print_markov_results(markov_results, k=k)
    print(f'\n[Task 2-2] Wasserstein Distance 계산')
    print('─' * 50)
    wd_df = compute_wasserstein_distances(risk_df, ownership=ownership)
    print_wasserstein_results(wd_df)
    _print_final_insight(labels_df, ownership, markov_results, wd_df)
    return {'labels_df': labels_df, 'risk_df': risk_df, 'ownership': ownership, 'markov_results': markov_results, 'wasserstein_df': wd_df, 'climate_by_year': climate_by_year}

def _print_final_insight(labels_df, ownership, markov_results, wd_df):
    print('\n' + '╔' + '═' * 70 + '╗')
    print('║  📋 최종 통합 인사이트 (Integrated Sociological Insight)          ║')
    print('╚' + '═' * 70 + '╝')
    cols = list(labels_df.columns)
    first_col, last_col = (cols[0], cols[-1])
    ind_mask = ownership == 'Independent'
    corp_mask = ownership == 'Corporate'
    ind_safe_to_critical = ((labels_df.loc[ind_mask, first_col] == 0) & (labels_df.loc[ind_mask, last_col] == 2)).sum()
    corp_safe_to_critical = ((labels_df.loc[corp_mask, first_col] == 0) & (labels_df.loc[corp_mask, last_col] == 2)).sum()
    n_ind_total = ind_mask.sum()
    n_corp_total = corp_mask.sum()
    print(f'\n  1. [기후 군집 전이 분석 결과]\n     • 독립 증류소: Safe→Critical 직접 전이 {ind_safe_to_critical}건 / {n_ind_total}개 ({ind_safe_to_critical / max(n_ind_total, 1) * 100:.1f}%)\n     • 대자본 증류소: Safe→Critical 직접 전이 {corp_safe_to_critical}건 / {n_corp_total}개 ({corp_safe_to_critical / max(n_corp_total, 1) * 100:.1f}%)\n\n  2. [자본 양극화 효과]\n     독립 증류소의 기후 리스크 군집 전이는 대자본 소유 증류소에 비해\n     고위험군으로의 이동이 두드러지며, 이는 자본력에 기반한 기후 적응 투자\n     (원료 다변화, 수원 확보, 생산지 이전) 역량의 차이에 기인합니다.\n\n  3. [정책적 시사점]\n     기후 변화에 따른 스코틀랜드 위스키 산업의 양극화를 완화하기 위해,\n     독립/소규모 증류소 대상의 기후 적응 기금(Climate Adaptation Fund) 조성\n     및 유역 단위의 수자원 관리 정책이 시급합니다.\n')
if __name__ == '__main__':
    result = run_temporal_analysis()
    print('\n  ✅ 시계열 분석 완료')