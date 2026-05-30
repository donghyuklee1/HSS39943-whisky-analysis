import pandas as pd
import sys
import os
import warnings
import time
warnings.filterwarnings('ignore')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from config import OUTPUT_DIR, RANDOM_SEED
from data.downloader import DataDownloader
from data.distillery_data import get_distillery_geodataframe
from data.preprocessing import SpatialProcessor
from models.clustering import ClimateClusterAnalyzer
from visualization.cluster_plots import ClusterPlotter
from visualization.folium_map import WhiskyClimateMap

def print_banner():
    banner = '\n╔══════════════════════════════════════════════════════════════════╗\n║                                                                  ║\n║   🥃  스코틀랜드 위스키 산업 기후 취약성 분석 파이프라인  🌍     ║\n║                                                                  ║\n║   Climate Vulnerability Analysis for Scottish Whisky Industry    ║\n║   ─ Water Stress × Barley Suitability × Regional Economy ─      ║\n║                                                                  ║\n╚══════════════════════════════════════════════════════════════════╝\n    '
    print(banner)

def print_insight(gdf):
    n_total = len(gdf)
    n_changed = gdf['cluster_changed'].sum()
    mean_delta_precip = gdf['delta_precip'].mean()
    high_risk_mask = gdf['risk_level'].str.contains('Critical|Warning', na=False)
    high_risk_distilleries = gdf.loc[high_risk_mask, 'name'].tolist()
    insight = f'\n╔══════════════════════════════════════════════════════════════════╗\n║                    📊  분석 결론 (Insight)                       ║\n╚══════════════════════════════════════════════════════════════════╝\n\n1. [기후-지형적 불평등] 2050년 기후 시나리오(CMIP6) 및 실제 지형 분석 결과, 스코틀랜드 {n_total}개\n   증류소 중 {n_changed}개({n_changed / n_total * 100:.1f}%)가 현재 기후/맛 군집에서\n   이탈하여 새로운 위험 군집으로 편입될 것으로 예측됩니다. 스코틀랜드 전역의\n   연평균 강수량은 약 {mean_delta_precip:+.1f}mm 변화할 것으로 분석되어, \n   기후 변화의 영향이 지역 유역(Catchment)별로 불균등하게 분배됨을 확인했습니다.\n\n2. [자본 양극화와 테루아] 고위험(Critical/Warning) 군집에 속하는 증류소들\n   ({', '.join(high_risk_distilleries[:8])} 등)은 주로 특정 유역의 수원(水源)과 \n   마이크로 클라이밋에 의존하는 테루아 기반 생산 방식을 고수합니다. \n   이들은 대기업 소유 증류소와 달리 생산지 이전이나 원료 대체가 불가능하여, \n   기후 위기가 곧 생존 위기로 직결됩니다.\n\n3. [데이터 기반 기후 적응] 본 분석은 Kaggle의 실제 위스키 맛 프로필 데이터와 \n   WorldClim 2.1 고해상도 기후 래스터, 그리고 실제 영국 수계 지형 데이터를\n   결합하여 군집 단위의 취약성을 식별했습니다. 이를 통해 어떤 맛을 가진\n   증류소들이 기후 변화에 가장 노출되어 있는지 정량적으로 파악할 수 있습니다.\n'
    print(insight)

def print_migration_matrix(gdf, optimal_k: int):
    print('\n  ┌─ 군집 이동 교차표 (Base → Future) ─────────────┐')
    cross = pd.crosstab(gdf['base_cluster'], gdf['future_cluster'], margins=True, margins_name='합계')
    col_labels = [f'F{c}' for c in range(optimal_k)] + ['합계']
    header = '  │  Base\\Future │ ' + ' │ '.join((f'{c:>5s}' for c in col_labels)) + ' │'
    print('  │' + '─' * (len(header) - 4) + '│')
    print(header)
    print('  │' + '─' * (len(header) - 4) + '│')
    for base_c in list(range(optimal_k)) + ['합계']:
        row_label = f'B{base_c}' if isinstance(base_c, int) else '합계'
        vals = []
        for fut_c in list(range(optimal_k)) + ['합계']:
            try:
                v = cross.loc[base_c, fut_c]
            except KeyError:
                v = 0
            vals.append(f'{v:>5d}')
        row = f'  │  {row_label:>11s} │ ' + ' │ '.join(vals) + ' │'
        print(row)
    print('  └' + '─' * (len(header) - 4) + '┘')
    stayed = (gdf['base_cluster'] == gdf['future_cluster']).sum()
    moved = len(gdf) - stayed
    print(f'\n  → 잔류: {stayed}개 ({stayed / len(gdf) * 100:.1f}%)  |  이동: {moved}개 ({moved / len(gdf) * 100:.1f}%)')

def print_region_vulnerability(gdf):
    print('\n  ┌─ 유역(Catchment)별 기후 취약성 요약 ──────────────────────────────────────────┐')
    header = f'  │ {'유역(Catchment)':>15s} │ {'증류소':>4s} │ {'Δ강수량(mm)':>11s} │ {'이동률':>6s} │ {'고위험':>4s} │'
    print(header)
    print('  │' + '─' * (len(header) - 4) + '│')
    for region in sorted(gdf['Catchment'].astype(str).unique()):
        mask = gdf['Catchment'] == region
        subset = gdf[mask]
        n = len(subset)
        avg_dp = subset['delta_precip'].mean()
        pct_changed = subset['cluster_changed'].mean() * 100
        n_high = subset['risk_level'].str.contains('Critical|Warning', na=False).sum()
        row = f'  │ {region[:15]:>15s} │ {n:>4d} │ {avg_dp:>+11.1f} │ {pct_changed:>5.1f}% │ {n_high:>4d} │'
        print(row)
    print('  └' + '─' * (len(header) - 4) + '┘')

def main():
    start_time = time.time()
    print_banner()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        print('\n[Step 1] 데이터 수집 (Kaggle, WorldClim, Catchments)')
        print('=' * 50)
        dl = DataDownloader()
        _ = dl.get_whisky_csv()
        gdf = get_distillery_geodataframe()
        raster_paths = dl.get_worldclim_data()
        catchments = dl.get_sepa_catchments()
        processor = SpatialProcessor()
        gdf = processor.process_all(gdf, raster_paths, catchments)
        analyzer = ClimateClusterAnalyzer()
        gdf = analyzer.fit(gdf)
        gdf = analyzer.predict_future(gdf)
        gdf = analyzer.assign_risk_labels(gdf)
        print('\n[Step 4] 분석 차트 생성')
        print('=' * 50)
        plotter = ClusterPlotter()
        plotter.plot_elbow_silhouette(analyzer.inertias, analyzer.silhouette_scores, analyzer.optimal_k)
        plotter.plot_cluster_comparison(gdf, analyzer.BASE_FEATURES, analyzer.FUTURE_FEATURES, analyzer.optimal_k)
        map_builder = WhiskyClimateMap()
        map_path = map_builder.build(gdf, analyzer.base_centroids_geo, analyzer.future_centroids_geo, analyzer.optimal_k)
        print('\n[Step 6] 군집 이동 분석 및 지역별 취약성')
        print('=' * 50)
        print_migration_matrix(gdf, analyzer.optimal_k)
        print_region_vulnerability(gdf)
        elapsed = time.time() - start_time
        print(f'\n{'=' * 60}')
        print(f'  ✅ 파이프라인 실행 완료 (소요 시간: {elapsed:.1f}초)')
        print(f'{'=' * 60}')
        print('\n  📁 생성된 파일:')
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f'     ├─ {f} ({size_kb:.1f} KB)')
        print(f'\n  📊 데이터 요약 (상위 10개 증류소):')
        summary_cols = ['name', 'Catchment', 'base_cluster', 'future_cluster', 'risk_level', 'delta_precip']
        print(gdf[summary_cols].head(10).to_string(index=False))
        print_insight(gdf)
        return gdf
    except FileNotFoundError as e:
        print(f'\n❌ 파일 오류: {e}')
        sys.exit(1)
    except RuntimeError as e:
        print(f'\n❌ 런타임 오류: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ 예상치 못한 오류: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()