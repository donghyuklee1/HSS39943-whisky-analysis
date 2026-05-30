import os
import sys
import time
import warnings
import traceback
warnings.filterwarnings('ignore')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from data.real_data_fetcher import get_real_distilleries_with_climate
from models.real_clustering import perform_real_clustering
from visualization.poster_map_real import create_poster_map_real

def print_section_header(title: str):
    print('\n' + '═' * 72)
    print(f'  {title}')
    print('═' * 72)

def main():
    start_time = time.time()
    print_section_header('Phase 1: Real Data Acquisition (Wikidata + Open-Meteo)')
    try:
        distilleries_gdf = get_real_distilleries_with_climate(limit=80)
    except Exception as e:
        print(f'❌ 데이터 수집 실패: {e}')
        traceback.print_exc()
        return
    print_section_header('Phase 2: Spatial Processing & K-Means Clustering')
    try:
        clustered_gdf = perform_real_clustering(distilleries_gdf)
    except Exception as e:
        print(f'❌ 클러스터링 실패: {e}')
        traceback.print_exc()
        return
    print_section_header('Phase 3: High-Fidelity Multi-Panel Visualization')
    try:
        poster_path = create_poster_map_real(clustered_gdf)
        print(f'  → 최종 결과물: {poster_path}')
    except Exception as e:
        print(f'❌ 시각화 실패: {e}')
        traceback.print_exc()
        return
    elapsed = time.time() - start_time
    print('\n' + '=' * 72)
    print(f'  🎉 100% Real Data Pipeline 실행 완료 (총 소요 시간: {elapsed:.1f}초)')
    print('=' * 72)
if __name__ == '__main__':
    main()