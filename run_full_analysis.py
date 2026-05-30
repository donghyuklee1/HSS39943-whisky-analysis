import os
import sys
import time
import warnings
import traceback
warnings.filterwarnings('ignore')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from config import OUTPUT_DIR
from main import main as legacy_main
from visualization.poster_map import create_poster_map
from models.temporal_analysis import run_temporal_analysis
from visualization.temporal_plots import run_temporal_visualizations

def print_section_header(title: str):
    print('\n' + '═' * 72)
    print(f'  {title}')
    print('═' * 72)

def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_section_header('Phase 1: 기존 공간 분석 파이프라인 실행 (main.py)')
    try:
        print('>> legacy main() 호출 중...')
        print('>> (통합 실행 속도를 위해 legacy main은 생략하고 신규 파이프라인에 집중합니다.)')
    except Exception as e:
        print(f'⚠ Legacy Pipeline Error: {e}')
    print_section_header('Phase 2: 학술 포스터용 다중 패널 지도 생성 (1x3 Layout)')
    try:
        poster_path = create_poster_map()
        print(f'  → 포스터 생성 완료: {poster_path}')
    except Exception as e:
        print(f'❌ 포스터 생성 실패: {e}')
        traceback.print_exc()
    print_section_header('Phase 3: 시계열 확률적 기후 취약성 분석 (Stochastic Analysis)')
    try:
        analysis_results = run_temporal_analysis(method='linear')
        print(f'  → 시계열 분석 계산 완료.')
    except Exception as e:
        print(f'❌ 시계열 분석 실패: {e}')
        traceback.print_exc()
        analysis_results = None
    print_section_header('Phase 4: 시계열 동적 시각화 (Plotly Sankey & Matplotlib Ridgeline)')
    if analysis_results:
        try:
            viz_paths = run_temporal_visualizations(analysis_results)
            print(f'  → 시계열 시각화 생성 완료:')
            for k, p in viz_paths.items():
                print(f'     - {k}: {p}')
        except Exception as e:
            print(f'❌ 시각화 생성 실패: {e}')
            traceback.print_exc()
    elapsed = time.time() - start_time
    print('\n' + '=' * 72)
    print(f'  🎉 전체 통합 파이프라인 실행 완료 (총 소요 시간: {elapsed:.1f}초)')
    print('=' * 72)
if __name__ == '__main__':
    main()