import os
import numpy as np
import folium
from folium.plugins import MiniMap, Fullscreen
from math import radians, sin, cos, sqrt, atan2
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MAP_CENTER, MAP_ZOOM, MAP_TILES, OUTPUT_DIR, CLUSTER_COLORS

class WhiskyClimateMap:

    def __init__(self, output_dir: str=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.map = None

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def _create_base_map(self) -> folium.Map:
        m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles=MAP_TILES, attr='CartoDB')
        title_html = '\n        <div style="position: fixed; top: 10px; left: 50%;\n                    transform: translateX(-50%); z-index: 9999;\n                    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #1a1a2e 100%);\n                    padding: 14px 28px; border-radius: 12px;\n                    border: 1px solid rgba(241,196,15,0.25);\n                    box-shadow: 0 8px 32px rgba(0,0,0,0.6),\n                                inset 0 1px 0 rgba(255,255,255,0.05);">\n            <h3 style="margin: 0; color: #f1c40f; font-family: \'Segoe UI\', sans-serif;\n                       font-size: 17px; text-align: center; letter-spacing: 0.5px;\n                       text-shadow: 0 0 20px rgba(241,196,15,0.3);">\n                🥃 Scottish Whisky Distillery — Climate Risk Map (2050)\n            </h3>\n            <p style="margin: 4px 0 0 0; color: #95a5a6; font-size: 11px; text-align: center;\n                      font-family: \'Segoe UI\', sans-serif;">\n                Water Stress × Barley Suitability — K-Means Cluster Migration Analysis\n            </p>\n        </div>\n        '
        m.get_root().html.add_child(folium.Element(title_html))
        animated_css = '\n        <style>\n            @keyframes dash-flow {\n                to { stroke-dashoffset: -30; }\n            }\n            .centroid-arrow {\n                animation: dash-flow 1.5s linear infinite;\n            }\n        </style>\n        '
        m.get_root().html.add_child(folium.Element(animated_css))
        MiniMap(tile_layer=folium.TileLayer(tiles='CartoDB positron'), position='bottomleft', width=150, height=120, toggle_display=True).add_to(m)
        Fullscreen(position='topright', title='전체화면', title_cancel='축소').add_to(m)
        return m

    def _get_cluster_color(self, cluster_id: int) -> str:
        return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

    def _add_legend(self, m: folium.Map, gdf) -> None:
        legend_items = []
        if 'risk_level' in gdf.columns:
            for cluster_id in sorted(gdf['future_cluster'].unique()):
                mask = gdf['future_cluster'] == cluster_id
                risk = gdf.loc[mask, 'risk_level'].iloc[0]
                color = self._get_cluster_color(cluster_id)
                count = mask.sum()
                legend_items.append((color, f'Cluster {cluster_id}: {risk} ({count}개)'))
        n_total = len(gdf)
        n_changed = gdf['cluster_changed'].sum() if 'cluster_changed' in gdf.columns else 0
        pct_changed = n_changed / n_total * 100 if n_total > 0 else 0
        legend_html = f"""\n        <div style="position: fixed; bottom: 30px; right: 20px; z-index: 9999;\n                    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);\n                    padding: 16px 20px; border-radius: 12px;\n                    border: 1px solid rgba(241,196,15,0.2);\n                    box-shadow: 0 8px 32px rgba(0,0,0,0.6);\n                    font-family: 'Segoe UI', sans-serif;\n                    min-width: 220px;">\n            <h4 style="margin: 0 0 10px 0; color: #f1c40f; font-size: 13px;\n                       border-bottom: 1px solid rgba(255,255,255,0.1);\n                       padding-bottom: 8px;">\n                ◆ Future Risk Clusters (2050)\n            </h4>\n        """
        for color, label in legend_items:
            legend_html += f'\n            <p style="margin: 4px 0; color: #e0e0e0; font-size: 11px;">\n                <span style="display: inline-block; width: 12px; height: 12px;\n                             background: {color}; border-radius: 50%;\n                             margin-right: 8px; vertical-align: middle;\n                             box-shadow: 0 0 6px {color}40;"></span>\n                {label}\n            </p>\n            '
        legend_html += f'\n            <hr style="border: 0.5px solid rgba(255,255,255,0.1); margin: 10px 0;">\n            <p style="margin: 3px 0; color: #e74c3c; font-size: 10px;">\n                ⚠ 군집 이동 증류소: {n_changed}개 ({pct_changed:.0f}%)\n            </p>\n            <p style="margin: 3px 0; color: #95a5a6; font-size: 10px;">\n                ◯ 과거 중심 / ● 미래 중심\n            </p>\n            <p style="margin: 3px 0; color: #95a5a6; font-size: 10px;">\n                → 화살표: 중심 이동 벡터 (km)\n            </p>\n            <p style="margin: 3px 0; color: #95a5a6; font-size: 10px;">\n                ◇ 점선 마커: 군집 이동 증류소\n            </p>\n        </div>\n        '
        m.get_root().html.add_child(folium.Element(legend_html))

    @staticmethod
    def _svg_bar_chart(base_precip: float, future_precip: float, width: int=200, height: int=60) -> str:
        bar_max_w = width - 90
        bar_h = 16
        gap = 6

        def bar(y, val, color, label):
            norm_val = max(0, min(val / 2500.0, 1.0))
            bw = max(int(norm_val * bar_max_w), 2)
            return f'<text x="0" y="{y + 12}" fill="#bdc3c7" font-size="10">{label}</text><rect x="85" y="{y}" width="{bw}" height="{bar_h}" rx="3" fill="{color}" opacity="0.85"/><text x="{85 + bw + 4}" y="{y + 12}" fill="#e0e0e0" font-size="9">{val:.0f}mm</text>'
        y = 5
        svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        svg += bar(y, base_precip, '#3498db', 'Precip(Base)')
        y += bar_h + gap
        svg += bar(y, future_precip, '#e74c3c', 'Precip(2050)')
        svg += '</svg>'
        return svg

    def _add_distillery_markers(self, m: folium.Map, gdf, fg: folium.FeatureGroup) -> None:
        for _, row in gdf.iterrows():
            cluster_color = self._get_cluster_color(int(row['future_cluster']))
            changed = row.get('cluster_changed', False)
            dw_icon = '▲' if row['delta_precip'] > 0 else '▼'
            dw_color = '#2ecc71' if row['delta_precip'] > 0 else '#e74c3c'
            svg_chart = self._svg_bar_chart(row['base_precip'], row['future_precip'])
            migration_badge = ''
            if changed:
                migration_badge = f'<div style="background: #e74c3c22; border: 1px solid #e74c3c55; border-radius: 6px; padding: 4px 8px; margin-top: 6px; text-align: center;"><span style="color: #e74c3c; font-size: 11px; font-weight: bold;">⚠ Cluster {int(row['base_cluster'])} → {int(row['future_cluster'])}</span></div>'
            popup_html = f"""\n            <div style="font-family: 'Segoe UI', sans-serif; font-size: 12px;\n                        min-width: 240px; max-width: 280px;\n                        background: #0d1b2a; color: #e0e0e0;\n                        padding: 14px; border-radius: 10px;\n                        border: 1px solid {cluster_color}40;">\n                <div style="display: flex; align-items: center; margin-bottom: 8px;">\n                    <span style="font-size: 20px; margin-right: 8px;">🥃</span>\n                    <div>\n                        <b style="color: {cluster_color}; font-size: 15px;\n                                  text-shadow: 0 0 10px {cluster_color}30;">\n                            {row['name']}\n                        </b>\n                        <br>\n                        <span style="color: #95a5a6; font-size: 11px;">{row['Catchment']} Catchment</span>\n                    </div>\n                </div>\n                <hr style="margin: 6px 0; border: 0.5px solid rgba(255,255,255,0.1);">\n                <table style="width: 100%; font-size: 11px; border-collapse: collapse;">\n                    <tr>\n                        <td style="padding: 2px 0; color: #95a5a6;">위험도</td>\n                        <td style="padding: 2px 0; text-align: right;">\n                            <b style="color: {cluster_color};">{row.get('risk_level', 'N/A')}</b>\n                        </td>\n                    </tr>\n                    <tr>\n                        <td style="padding: 2px 0; color: #95a5a6;">군집 (Base/Future)</td>\n                        <td style="padding: 2px 0; text-align: right;">\n                            {int(row['base_cluster'])} → {int(row['future_cluster'])}\n                        </td>\n                    </tr>\n                    <tr>\n                        <td style="padding: 2px 0; color: #95a5a6;">Δ 연 강수량 (mm)</td>\n                        <td style="padding: 2px 0; text-align: right;">\n                            <span style="color: {dw_color};">\n                                {dw_icon} {row['delta_precip']:+.1f}\n                            </span>\n                        </td>\n                    </tr>\n                </table>\n                <hr style="margin: 8px 0 6px 0; border: 0.5px solid rgba(255,255,255,0.1);">\n                <p style="margin: 0 0 4px 0; color: #95a5a6; font-size: 10px;">\n                    Base vs Future 기후 변수 비교:\n                </p>\n                {svg_chart}\n                {migration_badge}\n            </div>\n            """
            tooltip_text = f'🥃 {row['name']} | {row.get('risk_level', '')}'
            radius = 9 if changed else 7
            weight = 3 if changed else 1
            dash_array = '5, 3' if changed else None
            folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=radius, color='white' if not changed else '#f1c40f', weight=weight, dash_array=dash_array, fill=True, fill_color=cluster_color, fill_opacity=0.85, tooltip=folium.Tooltip(tooltip_text), popup=folium.Popup(popup_html, max_width=300)).add_to(fg)

    def _add_centroid_arrows(self, m: folium.Map, base_centroids: np.ndarray, future_centroids: np.ndarray, optimal_k: int, fg: folium.FeatureGroup) -> None:
        for c in range(optimal_k):
            base_lat, base_lon = base_centroids[c]
            future_lat, future_lon = future_centroids[c]
            dist_km = self._haversine_km(base_lat, base_lon, future_lat, future_lon)
            coord_dist = np.sqrt((future_lat - base_lat) ** 2 + (future_lon - base_lon) ** 2)
            color = self._get_cluster_color(c)
            folium.CircleMarker(location=[base_lat, base_lon], radius=16, color=color, weight=2, fill=False, opacity=0.7, tooltip=folium.Tooltip(f'<b>Cluster {c}</b> — 과거 중심<br>({base_lat:.2f}°N, {base_lon:.2f}°E)')).add_to(fg)
            folium.Marker(location=[base_lat, base_lon], icon=folium.DivIcon(html=f'<div style="font-size:9px; color:{color}; font-weight:bold; text-align:center; width:40px; margin-left:-20px; margin-top:18px; text-shadow: 0 0 4px #000;">Past</div>', icon_size=(40, 20))).add_to(fg)
            folium.CircleMarker(location=[future_lat, future_lon], radius=16, color=color, weight=2, fill=True, fill_color=color, fill_opacity=0.35, tooltip=folium.Tooltip(f'<b>Cluster {c}</b> — 미래 중심 (2050)<br>({future_lat:.2f}°N, {future_lon:.2f}°E)<br>이동 거리: {dist_km:.1f} km')).add_to(fg)
            folium.Marker(location=[future_lat, future_lon], icon=folium.DivIcon(html=f'<div style="font-size:9px; color:{color}; font-weight:bold; text-align:center; width:50px; margin-left:-25px; margin-top:18px; text-shadow: 0 0 4px #000;">Future</div>', icon_size=(50, 20))).add_to(fg)
            if coord_dist > 0.01:
                arrow_points = self._compute_arrow_head(base_lat, base_lon, future_lat, future_lon)
                folium.PolyLine(locations=[[base_lat, base_lon], [future_lat, future_lon]], color=color, weight=3, opacity=0.8, dash_array='10, 6', tooltip=folium.Tooltip(f'<b>Cluster {c}</b> 이동 거리: {dist_km:.1f} km')).add_to(fg)
                folium.PolyLine(locations=arrow_points, color=color, weight=4, opacity=0.9).add_to(fg)
                mid_lat = (base_lat + future_lat) / 2
                mid_lon = (base_lon + future_lon) / 2
                folium.Marker(location=[mid_lat, mid_lon], icon=folium.DivIcon(html=f'<div style="font-size:10px; color:#f1c40f; font-weight:bold; background: rgba(13,27,42,0.8); padding: 2px 6px; border-radius: 4px; border: 1px solid {color}40; text-align: center; white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,0.5);">{dist_km:.1f} km</div>', icon_size=(60, 20), icon_anchor=(30, 10))).add_to(fg)

    def _compute_arrow_head(self, lat1: float, lon1: float, lat2: float, lon2: float, size: float=0.08) -> list:
        angle = np.arctan2(lat2 - lat1, lon2 - lon1)
        left_angle = angle + np.pi * 0.8
        right_angle = angle - np.pi * 0.8
        left_lat = lat2 + size * np.sin(left_angle)
        left_lon = lon2 + size * np.cos(left_angle)
        right_lat = lat2 + size * np.sin(right_angle)
        right_lon = lon2 + size * np.cos(right_angle)
        return [[left_lat, left_lon], [lat2, lon2], [right_lat, right_lon]]

    def build(self, gdf, base_centroids: np.ndarray, future_centroids: np.ndarray, optimal_k: int) -> str:
        print('\n[Step 5] Folium 인터랙티브 지도 생성')
        print('=' * 50)
        try:
            self.map = self._create_base_map()
            fg_distilleries = folium.FeatureGroup(name='🥃 증류소 마커 (Distilleries)', show=True)
            fg_centroids = folium.FeatureGroup(name='→ 군집 중심 이동 (Centroid Shift)', show=True)
            self._add_distillery_markers(self.map, gdf, fg_distilleries)
            print(f'  ✓ {len(gdf)}개 증류소 마커 추가 (Popup + Tooltip)')
            self._add_centroid_arrows(self.map, base_centroids, future_centroids, optimal_k, fg_centroids)
            print(f'  ✓ {optimal_k}개 군집 중심 이동 벡터 추가 (km 라벨 포함)')
            fg_distilleries.add_to(self.map)
            fg_centroids.add_to(self.map)
            folium.LayerControl(collapsed=False, position='topright').add_to(self.map)
            print('  ✓ LayerControl 추가 (레이어 토글)')
            self._add_legend(self.map, gdf)
            print('  ✓ 범례 오버레이 추가')
            print('  ✓ MiniMap + Fullscreen 플러그인 추가')
            filepath = os.path.join(self.output_dir, 'whisky_climate_map.html')
            self.map.save(filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f'  ✓ 지도 저장: {os.path.basename(filepath)} ({size_kb:.1f} KB)')
            return filepath
        except Exception as e:
            raise RuntimeError(f'Folium 지도 생성 실패: {e}') from e
if __name__ == '__main__':
    print('folium_map.py 모듈 — 독립 실행은 main.py를 통해 진행하세요.')