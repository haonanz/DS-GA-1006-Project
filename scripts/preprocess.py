
from rtree import index
from shapely.geometry import shape, Point
import gzip
import geojson
import numpy as np
import os
import pandas as pd


NUMERICAL_FEATURE = ['price', 'num_sqft', 'monthly_cost', 'building_num_units',
					 'topic1', 'topic2', 'topic3', 'topic4', 'topic5',
					 'building_comp_num_sqft', 'building_comp_price',
					 'neighborhood_comp_num_sqft', 'neighborhood_comp_price',
					 ]

CATEGORICAL_FEATURE = ['date', 'borough', 'status', 'type', 'built_date',
					   'num_beds', 'num_baths', 'neighborhood', 'community_district' ]

MULTICATEGORICAL_FEATURE = ['amenities_list', 'transit_list']


def clean_string(string):
	return '_'.join(string.encode('ascii', 'ignore').lower().replace('-', ' ').replace('/', ' ').replace(',', ' ').strip().split())

def clean_number(number):
	return number.replace(',', '').replace('$', '')


# convert string to float, replace nan with mean, and add an indicator feature for missing data
def handleNumericalColumn(df, col_name, processed_feature):
	df[col_name] = df[col_name].astype('float')
	if np.any(np.isnan(df[col_name])) == True:
		indicator_col = '{}_missing'.format(col_name)
		df.insert(df.columns.get_loc(col_name) + 1, indicator_col, np.isnan(df[col_name]).astype('int'))
		df[col_name].fillna(np.mean(df[col_name]), inplace=True)
		processed_feature.append(indicator_col)
		print "adding indicator column {} for missing data".format(col_name)
	processed_feature.append(col_name)

# create dummy columns for categorical features
def handleCategoricalColumn(df, col_name, processed_feature):
	# the get_dummies routine returns one feature for each categorical value (and one for null)
	dummies = pd.get_dummies(df[col_name], prefix=col_name, dummy_na=np.any(pd.isnull(df[col_name])))
	for dummy_col in dummies.columns[:-1]:
		df.insert(len(df.columns), dummy_col, dummies[dummy_col].astype('int'))
		processed_feature.append(dummy_col)
		print "adding dummy column {} for categorical feature".format(dummy_col)
	print "omiting dummy column {} for categorical feature".format(dummies.columns[-1])

# create dummy columns for multi-categorical features
def handleMultiCategoricalColumn(df, col_name, processed_feature):
	idx_list = []
	cat_list = []
	for index, entry in df[col_name].iteritems():
		if not pd.isnull(entry):
			categories = list(set(entry.split()))
			idx_list.extend([ index ] * len(categories))
			cat_list.extend(categories)
		else:
			idx_list.append(index)
			cat_list.append(np.nan)

	tmp_series = pd.Series(cat_list, index=idx_list)
	tmp_dummies = pd.get_dummies(tmp_series, prefix=col_name, dummy_na=np.any(pd.isnull(tmp_series)))
	dummies = tmp_dummies.groupby(tmp_dummies.index).sum()
	for dummy_col in dummies.columns[:-1]:
		df.insert(len(df.columns), dummy_col, dummies[dummy_col].astype('int'))
		processed_feature.append(dummy_col)
		print "adding dummy column {} for multi-categorical feature".format(dummy_col)
	print "omiting dummy column {} for multi-categorical feature".format(dummies.columns[-1])


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

def getGeoDataFromGPSString(index, index_data, gps_string):
	tokens = gps_string.split()
	assert len(tokens) == 2
	data = getGeoDataFromLongLat(index, index_data, float(tokens[1]), float(tokens[0]))
	if not pd.isnull(data):
		return data
	else:
		return getGeoDataFromLongLat(index, index_data, float(tokens[0]), float(tokens[1]))


def transform_features(df):
	# transform closing/sale date to year
	df.date = df.date.str[0:4]

	print 'transforming num_beds and num_baths'

	# tranform num_beds to something with more reasonable bins
	saleid = []
	num_beds_list = []
	df.insert(df.columns.get_loc('num_beds') + 1, 'num_beds_raw', df.num_beds)
	for index, entry in df.num_beds_raw.iteritems():
		if not pd.isnull(entry) and entry != 'studio':
			x = float(str(entry).replace('+', ''))
			saleid.append(index)
			num_beds_list.append(str(int(round(x))) if x < 10 else '10+')
		else:
			saleid.append(index)
			num_beds_list.append(entry)

	df.num_beds = pd.Series(num_beds_list, index=saleid)

	# same with num_baths
	saleid = []
	num_baths_list = []
	df.insert(df.columns.get_loc('num_baths') + 1, 'num_baths_raw', df.num_baths)
	for index, entry in df.num_baths_raw.iteritems():
		if not pd.isnull(entry):
			x = float(str(entry).replace('+', '.5').replace('.5.5', '.5'))
			saleid.append(index)
			num_baths_list.append(str(x) if x < 10 else '10+')
		else:
			saleid.append(index)
			num_baths_list.append(np.nan)

	df.num_baths = pd.Series(num_baths_list, index=saleid)

	print 'transforming built date'

	# transform building built date to prewar/postwar/modern
	df.insert(df.columns.get_loc('built_date') + 1, 'built_date_raw', df.built_date)
	df.loc[                               df.built_date_raw <= 1945 , 'built_date'] = 'prewar'
	df.loc[(1946 <= df.built_date_raw) & (df.built_date_raw <= 1999), 'built_date'] = 'postwar'
	df.loc[ 2000 <= df.built_date_raw                               , 'built_date'] = 'modern'

	print 'transforming transit list'

	# transform transit list format
	saleid = []
	transit_list = []
	df.insert(df.columns.get_loc('transit_list') + 1, 'transit_list_raw', df.transit_list)
	for index, entry in df.transit_list_raw.iteritems():
		if not pd.isnull(entry):
			transit_tokens = np.nan
			transit_raw_list = list(set(entry.split()))
			for item in transit_raw_list:
				tokens = item.split('@')
				if len(tokens) != 3: assert False
				if tokens[-1] == 'under_500_feet':
					distance = 'under_500_feet'
				else:
					distance_in_miles = float(tokens[-1].replace('_miles', '_mile').replace('_mile', ''))
					if distance_in_miles <= 0.25:
						distance = 'under_quarter_mile'
					elif distance_in_miles <= 0.5:
						distance = 'under_half_mile'
					else:
						distance = 'over_half_mile'
				transit_lines = tokens[0].split('%')
				new_token = ' '.join([x + '_' + distance for x in transit_lines])
				transit_tokens = transit_tokens + ' ' + new_token if not pd.isnull(transit_tokens) else new_token
			saleid.append(index)
			transit_list.append(transit_tokens)
		else:
			saleid.append(index)
			transit_list.append(np.nan)

	df.transit_list = pd.Series(transit_list, index=saleid)

	print 'creating new features based on comps in building and neighborhood'

	# figure out average sqft per building per bedroom type
	saleid = []
	bld_comp_sqft_list = []
	nbh_comp_sqft_list = []
	bld_comp_price_list = []
	nbh_comp_price_list = []
	bld_group_sum = df.groupby([df.num_beds, df.type, df.building_url]).sum()
	bld_group_mean = df.groupby([df.num_beds, df.type, df.building_url]).mean()
	bld_group_count = df.groupby([df.num_beds, df.type, df.building_url]).count()
	nbh_group_sum = df.groupby([df.num_beds, df.type, df.neighborhood]).sum()
	nbh_group_mean = df.groupby([df.num_beds, df.type, df.neighborhood]).mean()
	nbh_group_count = df.groupby([df.num_beds, df.type, df.neighborhood]).count()
	for index, row in df.iterrows():
		bld_group_sqft = np.nan
		nbh_group_sqft = np.nan
		bld_group_price = np.nan
		nbh_group_price = np.nan

		if not (pd.isnull(row.num_beds)  or pd.isnull(row.type) or pd.isnull(row.building_url)):
			bld_group_sqft = bld_group_mean.num_sqft[row.num_beds, row.type, row.building_url]
			bld_group_price_count = bld_group_count.price[row.num_beds, row.type, row.building_url]
			if bld_group_price_count > 1:
				bld_group_price = (bld_group_sum.price[row.num_beds, row.type, row.building_url] - row.price) / (bld_group_price_count - 1)

		if not (pd.isnull(row.num_beds) or pd.isnull(row.type) or pd.isnull(row.neighborhood)):
			nbh_group_sqft = nbh_group_mean.num_sqft[row.num_beds, row.type, row.neighborhood]
			nbh_group_price_count = nbh_group_count.price[row.num_beds, row.type, row.neighborhood]
			if nbh_group_price_count > 1:
				nbh_group_price = (nbh_group_sum.price[row.num_beds, row.type, row.neighborhood] - row.price) / (nbh_group_price_count - 1)

		saleid.append(index)
		bld_comp_sqft_list.append(bld_group_sqft)
		nbh_comp_sqft_list.append(nbh_group_sqft)
		bld_comp_price_list.append(bld_group_price)
		nbh_comp_price_list.append(nbh_group_price)

	df['building_comp_num_sqft'] = pd.Series(bld_comp_sqft_list, index=saleid)
	df['building_comp_price'] = pd.Series(bld_comp_price_list, index=saleid)
	df['neighborhood_comp_num_sqft'] = pd.Series(nbh_comp_sqft_list, index=saleid)
	df['neighborhood_comp_price'] = pd.Series(nbh_comp_price_list, index=saleid)


# read data frames and drop rows without GPS coordinate
df = pd.DataFrame()
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
for file in os.listdir(data_dir):
	if file.startswith('sales_listings') and file.endswith('csv.gz'):
		new_df = pd.read_csv(os.path.join(data_dir, file), header=0, index_col=0).dropna(subset=['gps_coordinates'], axis=0)
		df = df.append(new_df) if df.size > 0 else new_df
		assert(np.all(new_df.columns == df.columns))
assert df.size > 0

# join with LDA csv file
df = df.join(pd.read_csv(os.path.join(data_dir, 'lda_representation.csv.gz'), header=0, index_col=0))

# drop duplicate row indices
df = df[~df.index.duplicated()]

saleids = []
boroughs = []
neighborhoods = []
communities = []
borough_index, borough_data = createSpatialIndex(os.path.join(data_dir, 'nyc_boroughs.geojson'), 'borough')
community_index, community_data = createSpatialIndex(os.path.join(data_dir, 'nyc_communities.geojson'), 'communityDistrict', omit_list=["164"])
neigborhood_index, neighborhood_data = createSpatialIndex(os.path.join(data_dir, 'nyc_neighborhoods.geojson'), 'neighborhood', omit_list=["central_park"])
for index, gps_coordinates in df.gps_coordinates.iteritems():
	saleids.append(index)
	boroughs.append(getGeoDataFromGPSString(borough_index, borough_data, gps_coordinates))
	communities.append(getGeoDataFromGPSString(community_index, community_data, gps_coordinates))
	neighborhoods.append(getGeoDataFromGPSString(neigborhood_index, neighborhood_data, gps_coordinates))
	if pd.isnull(boroughs[-1]) or pd.isnull(communities[-1]) or pd.isnull(neighborhoods[-1]):
		print 'cannot map GPS coordinates to NYC zone!', index, gps_coordinates, \
				boroughs[-1], neighborhoods[-1], communities[-1], df.ix[index, 'url']
	# if count % 10000 == 0:
	# 	print count, index, gps_coordinates, df.ix[index, 'neighborhood'], \
	# 			boroughs[-1], neighborhoods[-1], communities[-1], df.ix[index, 'url']

df['borough'] = pd.Series(boroughs, index=saleids)
df['neighborhood'] = pd.Series(neighborhoods, index=saleids)
df['community_district'] = pd.Series(communities, index=saleids)

# drop records with weird GPS coordinates
df = df.dropna(subset=['borough', 'neighborhood', 'community_district'], axis=0)

# load the test saleids; we are done dropping rows at this point
test_id_set = set()
with gzip.open(os.path.join(data_dir, 'test_saleids.txt.gz'), 'r') as fout:
	for line in fout:
		test_id_set.add(line.strip())

train_id_set = set()
with gzip.open(os.path.join(data_dir, 'train_saleids.txt.gz'), 'r') as fout:
	for line in fout:
		train_id_set.add(line.strip())

# get train and test ids
train_test_intersection = train_id_set.intersection(test_id_set)
# assert not train_test_intersection, 'Test and Train saleid lists are not disjoint!'

test_ids = []
train_ids = []
for saleid in df.index:
	if saleid in train_test_intersection:
		continue
	elif saleid in test_id_set:
		test_ids.append(saleid)
	elif saleid in train_id_set:
		train_ids.append(saleid)

# remove all data outside of the train and test set
df = df.ix[test_ids + train_ids]

print 'found %d out of %d keys in data frame for training' % (len(train_ids), len(train_id_set))
print 'found %d out of %d keys in data frame for testing' % (len(test_ids), len(test_id_set))

# transform features
transform_features(df)

# preprocess the features
processed_feature = []

for col_name in NUMERICAL_FEATURE:
	print "processing numerical feature: {}".format(col_name)
	assert col_name in df.columns
	handleNumericalColumn(df, col_name, processed_feature)

for col_name in CATEGORICAL_FEATURE:
	print "processing categorical feature: {}".format(col_name)
	assert col_name in df.columns
	handleCategoricalColumn(df, col_name, processed_feature)

for col_name in MULTICATEGORICAL_FEATURE:
	print "processing multi categorical feature: {}".format(col_name)
	assert col_name in df.columns
	handleMultiCategoricalColumn(df, col_name, processed_feature)

# check for NaN in the processed data frame
for col_name in processed_feature:
	assert np.any(pd.isnull(df[col_name])) == False, 'Column {} still has NaN entries!'.format(col_name)

# write train and data to a csv file
df.ix[train_ids].to_csv(os.path.join(data_dir, 'features.train.csv.gz'), compression='gzip', columns=processed_feature)
df.ix[ test_ids].to_csv(os.path.join(data_dir, 'features.test.csv.gz'), compression='gzip', columns=processed_feature)
