import os
import pandas as pd
import numpy as np

#Read data
print 'Read in featurized data...'
cols_to_read = ['saleid',
                'price',
                'num_sqft',
                'topic1',
                'topic2',
                'topic3',
                'topic4',
                'topic5',
                'topic6',
                'topic7',
                'topic8',
                'topic9',
                'topic10']

map_set = pd.read_csv('./data/features.train.csv.gz',
                    index_col = 0,
                    usecols=cols_to_read,
                    compression='gzip')

print 'Get lat/long from raw data...'
# read data frames and drop rows without GPS coordinate
df = pd.DataFrame()
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
for file in os.listdir(data_dir):
    if file.startswith('sales_listings') and file.endswith('csv.gz'):
        new_df = pd.read_csv(os.path.join(data_dir, file), header=0, index_col=0, usecols=['saleid','gps_coordinates']).dropna(subset=['gps_coordinates'], axis=0)
        df = df.append(new_df) if df.size > 0 else new_df
        assert(np.all(new_df.columns == df.columns))
assert df.size > 0

print 'Join feature subset and lat/long...'

def split_gps(gps_coordinates):
    lattitude, longitude = gps_coordinates.split(' ')
    lattitude = float(lattitude)
    longitude = float(longitude)
    return lattitude, longitude

# Add lat/long features

df['latitude'],df['longitude'] = zip(*df['gps_coordinates'].map(split_gps))
del df['gps_coordinates']

#Join map_set
map_set = map_set.join(df)
print 'Number of null values from join...'
print np.sum(pd.isnull(map_set))

print 'Writing data to disk for cartodb...'
map_set.to_csv('./data/cartodb.features.csv.gz', compression='gzip', index=True)
