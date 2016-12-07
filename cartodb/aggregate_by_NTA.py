from rtree import index
from shapely.geometry import shape, Point
import gzip
import geojson
import numpy as np
import os
import pandas as pd

def clean_string(string):
    return '_'.join(string.encode('ascii', 'ignore').lower().replace('-', ' ').replace('/', ' ').replace(',', ' ').strip().split())

def createSpatialIndex(geojson_path, data_name, omit_list=[]):
    index_data = []
    spatial_index = index.Index()
    with open(geojson_path, 'r') as geojson_obj:
        geojson_obj = geojson.load(geojson_obj)
        for feature in geojson_obj['features']:
            data_value = clean_string(str(feature['properties'][data_name]))
            if data_value not in omit_list:
                geometry = shape(feature['geometry']).buffer(0.001)
                spatial_index.insert(len(index_data), geometry.bounds)
                index_data.append((data_value, geometry))
    return spatial_index, index_data

def getGeoDataFromLongLat(index, index_data, longitude, latitude):
    point = Point(longitude, latitude)
    candidates = index.intersection((point.x, point.y, point.x, point.y))
    best_data = np.nan
    best_distance = None
    for candidate in candidates:
        if index_data[candidate][1].contains(point):
            distance = index_data[candidate][1].distance(point)
            if not best_distance or distance < best_distance:
                best_data = index_data[candidate][0]
                best_distance = distance
    return best_data

print 'Getting neighborhood shapefile...'

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
neigborhood_index, neighborhood_data = createSpatialIndex(os.path.join(data_dir, 'nyc_neighborhoods.geojson'), 'neighborhood', omit_list=["central_park"])

print 'Reading in data...'
df = pd.read_csv('./data/cartodb.features.csv.gz',
                    header = 0,
                    index_col = 0,
                    compression='gzip')

print 'Running neighborhood point in polygon test...'
df['neighborhood'] = df.apply(lambda x: getGeoDataFromLongLat(neigborhood_index, neighborhood_data, x['longitude'], x['latitude']), axis=1)

print 'Grouping by neighborhood...'
groupby_object = df.groupby('neighborhood')

groupby_object = groupby_object.agg('median')\
                               .rename(columns = lambda x: x + ' median')\
                               .join(pd.DataFrame(groupby_object.size(),
                                                  columns=['counts']))

print 'Adding aggregate features to geojson...'
with open(os.path.join(data_dir, 'nyc_neighborhoods.geojson'), 'r') as geojson_obj:
    geojson_obj = geojson.load(geojson_obj)
    for index, cols in groupby_object.iterrows():
        for feature in geojson_obj['features']:
            if clean_string(feature['properties']['neighborhood']) == index:
                feature['properties'].update(cols.to_dict())
                break
            else:
                if feature == geojson_obj['features'][-1]:
                    print "Couldn't find feature for {}".format(index)

with open(os.path.join(data_dir, 'nyc_neighborhoods_with_medians.geojson'), 'w') as outfile:
    geojson.dump(geojson_obj, outfile)
