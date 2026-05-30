import os
import time
import traceback
from data.real_data_fetcher import get_real_distilleries_with_climate
from models.gwr_analysis import run_gwr_analysis
from visualization.gwr_map import draw_gwr_map

def main():
    print('========================================================================')
    print('  지리적 가중 회귀(GWR) 기반 스코틀랜드 위스키 기후 위기 분석 파이프라인')
    print('========================================================================')
    start_time = time.time()
    try:
        distilleries_gdf = get_real_distilleries_with_climate(limit=80)
        distilleries_gdf = distilleries_gdf.dropna(subset=['temp_increase', 'precip_change_pct', 'humidity_drop'])
        if len(distilleries_gdf) < 10:
            print('⚠ 기후 데이터를 성공적으로 수집한 증류소가 부족하여 GWR 분석을 진행할 수 없습니다.')
            return
        print(f'\n총 {len(distilleries_gdf)}개의 증류소 데이터로 GWR 분석을 시작합니다.\n')
        gwr_gdf, ols_r2, gwr_r2 = run_gwr_analysis(distilleries_gdf)
        draw_gwr_map(gwr_gdf, output_path='output/gwr_sensitivity_map.png')
    except Exception as e:
        print(f'❌ 파이프라인 실행 중 오류 발생: {e}')
        traceback.print_exc()
    elapsed = time.time() - start_time
    print(f'\n========================================================================')
    print(f'  🎉 GWR Pipeline 실행 완료 (총 소요 시간: {elapsed:.1f}초)')
    print(f'========================================================================')
if __name__ == '__main__':
    main()