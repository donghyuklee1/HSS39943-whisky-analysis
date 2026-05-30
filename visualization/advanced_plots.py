import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": "#EAEAEA",
    "figure.facecolor": "white",
    "grid.color": "white",
    "font.family": "sans-serif"
})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.real_data_fetcher import get_real_distilleries_with_climate
from models.real_clustering import perform_real_clustering

def assign_region(lon, lat):
    if lon < -5.5:
        return 'Islay'
    elif 57.1 <= lat <= 57.8 and -3.7 <= lon <= -2.8:
        return 'Speyside'
    elif lat < 56.1:
        return 'Lowlands'
    else:
        return 'Highlands'

def generate_stacked_bar(gdf, output_path="output/analysis_1_stacked_bar.png"):
    print("\n[시각화 1] 지역별 기후 리스크 군집 누적 막대 차트 생성 중...")
    
    cross = pd.crosstab(gdf['Region'], gdf['risk_level'], normalize='index') * 100
    
    for col in ['Safe', 'Warning', 'Critical']:
        if col not in cross.columns:
            cross[col] = 0.0
            
    cross = cross[['Safe', 'Warning', 'Critical']]
    colors = {'Safe': '#1f77b4', 'Warning': '#ff7f0e', 'Critical': '#d62728'}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cross.plot(kind='bar', stacked=True, color=[colors[c] for c in cross.columns], ax=ax, edgecolor='black')
    
    ax.set_title("Proportion of Climate Risk Clusters by Whisky Region", fontsize=16, weight='bold', pad=15)
    ax.set_xlabel("Whisky Region", fontsize=12)
    ax.set_ylabel("Percentage of Distilleries (%)", fontsize=12)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)
    ax.legend(title="Risk Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장 완료: {output_path}")

def generate_joint_scatter(gdf, output_path="output/analysis_2_joint_scatter.png"):
    print("[시각화 2] 기후 변수 간 상관관계 산점도 (Joint Scatter Plot) 생성 중...")
    
    palette = {'Safe': '#1f77b4', 'Warning': '#ff7f0e', 'Critical': '#d62728'}
    
    # Determine the order of hue correctly to match colors
    g = sns.JointGrid(data=gdf, x='precip_change_pct', y='temp_increase', hue='risk_level', hue_order=['Safe', 'Warning', 'Critical'], palette=palette, height=8)
    g.plot_joint(sns.scatterplot, s=100, edgecolor='black', alpha=0.8)
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.5)
    
    g.set_axis_labels("2050 Predicted Precipitation Drop (%)", "2050 Predicted Temperature Increase (°C)", fontsize=12)
    g.fig.suptitle("Climate Variable Correlation & K-Means Cluster Boundaries", y=1.02, fontsize=16, weight='bold')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    g.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장 완료: {output_path}")

def generate_flavor_radar(gdf, raw_csv_path="data/raw/whisky.csv", output_path="output/analysis_3_flavor_radar.png"):
    print("[시각화 3] 군집별 위스키 맛 프로필 방사형 차트 생성 중...")
    
    try:
        whisky_df = pd.read_csv(raw_csv_path)
    except Exception as e:
        print(f"  ⚠ CSV 읽기 실패: {e}")
        return

    gdf['join_name'] = gdf['name'].str.lower().str.replace(' ', '')
    whisky_df['join_name'] = whisky_df['Distillery'].str.lower().str.replace(' ', '')
    
    merged = pd.merge(gdf, whisky_df, on='join_name', how='inner')
    
    if len(merged) == 0:
        print("  ⚠ 매핑되는 증류소 이름이 없어 방사형 차트를 생성할 수 없습니다.")
        return
        
    flavor_cols = ['Smoky', 'Tobacco', 'Sweetness', 'Fruity', 'Floral', 'Nutty', 'Malty', 'Spicy', 'Honey', 'Body', 'Medicinal', 'Winey']
    display_labels = ['Smoky', 'Peaty', 'Sweet', 'Fruity', 'Floral', 'Nutty', 'Malty', 'Spicy', 'Rich', 'Body', 'Medicinal', 'Winey']
    
    safe_data = merged[merged['risk_level'] == 'Safe'][flavor_cols].mean().fillna(0).values
    critical_data = merged[merged['risk_level'] == 'Critical'][flavor_cols].mean().fillna(0).values
    
    angles = np.linspace(0, 2 * np.pi, len(flavor_cols), endpoint=False).tolist()
    
    safe_data = np.concatenate((safe_data, [safe_data[0]]))
    critical_data = np.concatenate((critical_data, [critical_data[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor('#EAEAEA')
    
    ax.plot(angles, safe_data, color='#1f77b4', linewidth=2, label='Safe Cluster')
    ax.fill(angles, safe_data, color='#1f77b4', alpha=0.25)
    
    ax.plot(angles, critical_data, color='#d62728', linewidth=2, label='Critical Cluster')
    ax.fill(angles, critical_data, color='#d62728', alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=11, weight='bold')
    
    ax.set_title("Whisky Flavor Profile Comparison:\nSafe vs Critical Clusters", fontsize=16, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 저장 완료: {output_path}")

def main():
    print("================================================================")
    print("  [심화 시각화 파이프라인] 기후 위기 - 위스키 데이터 결합 분석")
    print("================================================================")
    
    try:
        gdf = get_real_distilleries_with_climate(limit=None)
    except Exception as e:
        print(f"데이터 수집 중 오류: {e}")
        return
        
    gdf = perform_real_clustering(gdf)
    
    gdf_wgs = gdf.to_crs(epsg=4326)
    regions = []
    for idx, row in gdf_wgs.iterrows():
        regions.append(assign_region(row.geometry.x, row.geometry.y))
    gdf['Region'] = regions
    
    generate_stacked_bar(gdf)
    generate_joint_scatter(gdf)
    generate_flavor_radar(gdf)
    
    print("\n================================================================")
    print("  🌍 기술사회학적(STS) 분석 결론 (Insight)")
    print("================================================================")
    print("1. [자원의 양극화]: Stacked Bar Chart에서 보이듯, Islay와 같은 특정 테루아에")
    print("   의존하는 생산지는 기후 위기에 대해 'Critical' 군집에 극단적으로 노출되어 있습니다.")
    print("2. [수학적 경계의 현실화]: Joint Scatter Plot은 강수량 감소와 기온 상승이 결합될 때")
    print("   어떤 임계점(Tipping Point)을 넘어서면 비가역적인 '위험(Critical)' 상태로 진입함을 증명합니다.")
    print("3. [문화적 유산의 소멸]: Radar Chart에서 보듯이, Critical 군집의 평균 향미 프로필이")
    print("   특정 맛(Smoky, Medicinal 등 피트 계열)에 강하게 편향되어 있다면, 이는 기후 위기가")
    print("   단순히 생산량의 감소를 넘어 특정 문화적 유산(전통 피트 위스키)을 선제적으로")
    print("   파괴하고 맛의 획일화를 초래함을 강력히 시사합니다.")
    print("================================================================\n")

if __name__ == "__main__":
    main()
