import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
CRS_WGS84 = 'EPSG:4326'
SCOTLAND_BOUNDS = {'lon_min': -8.0, 'lon_max': -1.0, 'lat_min': 54.5, 'lat_max': 59.0}
RASTER_RESOLUTION = 0.01
RASTER_WIDTH = int((SCOTLAND_BOUNDS['lon_max'] - SCOTLAND_BOUNDS['lon_min']) / RASTER_RESOLUTION)
RASTER_HEIGHT = int((SCOTLAND_BOUNDS['lat_max'] - SCOTLAND_BOUNDS['lat_min']) / RASTER_RESOLUTION)
RANDOM_SEED = 42
K_RANGE = range(2, 11)
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10
MAP_CENTER = [56.8, -4.5]
MAP_ZOOM = 7
MAP_TILES = 'CartoDB dark_matter'
CLUSTER_COLORS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c', '#e84393', '#fd79a8']