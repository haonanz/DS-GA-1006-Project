import os
import pandas as pd
import numpy as np
import gzip


NUMERICAL_FEATURE = ['price', 'num_sqft', 'monthly_cost', 'building_num_units',
					 'topic1', 'topic2', 'topic3', 'topic4', 'topic5'
					 ]

CATEGORICAL_FEATURE = ['date', 'borough', 'status', 'type', 'built_date', 'num_sqft_heuristic',
					   'num_beds', 'num_baths', 'neighborhood', 'school_district'
					   ]

MULTICATEGORICAL_FEATURE = ['amenities_list', 'transit_list']


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

	print 'transforming num_sqft'

	# figure out average sqft per building per bedroom type
	saleid = []
	num_sqft_list = []
	num_sqft_heuristic_list = []
	df.insert(df.columns.get_loc('num_sqft') + 1, 'num_sqft_raw', df.num_sqft)
	df.insert(df.columns.get_loc('num_sqft') + 1, 'num_sqft_heuristic', np.zeros(len(df.index)))
	grouping1 = df.num_sqft.groupby([df.num_beds, df.building_url]).mean()
	grouping2 = df.num_sqft.groupby([df.num_beds, df.neighborhood]).mean()
	for index, row in df.iterrows():
		heuristic = 0
		entry = row.num_sqft
		if pd.isnull(entry) and not (pd.isnull(row.num_beds) or pd.isnull(row.building_url)):
			if not np.isnan(grouping1[row.num_beds, row.building_url]):
				entry = grouping1[row.num_beds, row.building_url]
				heuristic = 1
		if pd.isnull(entry) and not (pd.isnull(row.num_beds) or pd.isnull(row.neighborhood)):
			if not np.isnan(grouping2[row.num_beds, row.neighborhood]):
				entry = grouping2[row.num_beds, row.neighborhood]
				heuristic = 2
		saleid.append(index)
		num_sqft_list.append(entry)
		num_sqft_heuristic_list.append(heuristic)

	df.num_sqft = pd.Series(num_sqft_list, index=saleid)
	df.num_sqft_heuristic = pd.Series(num_sqft_heuristic_list, index=saleid)


# read data frame, drop rows without GPS coordinate
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
df1 = pd.read_csv(os.path.join(data_dir, 'sales_listings.1.csv.gz'), header=0, index_col=0).dropna(subset=['gps_coordinates'], axis=0)
df2 = pd.read_csv(os.path.join(data_dir, 'sales_listings.2.csv.gz'), header=0, index_col=0).dropna(subset=['gps_coordinates'], axis=0)
df3 = pd.read_csv(os.path.join(data_dir, 'sales_listings.3.csv.gz'), header=0, index_col=0).dropna(subset=['gps_coordinates'], axis=0)
df4 = pd.read_csv(os.path.join(data_dir, 'sales_listings.4.csv.gz'), header=0, index_col=0).dropna(subset=['gps_coordinates'], axis=0)
lda = pd.read_csv(os.path.join(data_dir, 'lda_representation.csv.gz'), header=0, index_col=0)

assert(np.all(df1.columns == df2.columns))
assert(np.all(df1.columns == df3.columns))
assert(np.all(df1.columns == df4.columns))

# append the dataframes and join with lda csv; drop duplicate row indices
df = df1.append(df2).append(df3).append(df4).join(lda)
df = df[~df.index.duplicated()]

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

# load the test saleids; we are done dropping rows at this point
test_id_set = set()
with gzip.open(os.path.join(data_dir, 'test_saleids.txt.gz'), 'r') as fout:
	for line in fout:
		test_id_set.add(line.strip())

train_id_set = set()
with gzip.open(os.path.join(data_dir, 'train_saleids.txt.gz'), 'r') as fout:
	for line in fout:
		train_id_set.add(line.strip())

train_test_intersection = train_id_set.intersection(test_id_set)
# assert not train_test_intersection, 'Test and Train saleid lists are not disjoint!'

test_ids = []
train_ids = []
for saleid in df.index:
	if saleid in train_test_intersection:
		continue
	elif saleid in test_id_set:
		test_ids.append(saleid)
	else:
		train_ids.append(saleid)

# write train and data to a csv file
df.ix[train_ids].to_csv(os.path.join(data_dir, 'features.train.csv.gz'), compression='gzip', columns=processed_feature)
df.ix[ test_ids].to_csv(os.path.join(data_dir, 'features.test.csv.gz'), compression='gzip', columns=processed_feature)
