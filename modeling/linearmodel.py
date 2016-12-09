from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.linear_model import ElasticNetCV, ElasticNet
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def get_X_dataframe(df):
	good_features = []
	for column in df:
		if 'num_sqft' == column:
			good_features.append(column)
		if 'topic' in column:
			good_features.append(column)
		if 'num_beds' in column or 'num_baths' in column:
			good_features.append(column)
		if 'amenities_list' in column:
			good_features.append(column)
		if 'transit_list' in column:
			good_features.append(column)
		if 'type' in column:
			good_features.append(column)
		if 'community_district' in column:
			good_features.append(column)
		if 'neighborhood' in column:
			good_features.append(column)
		if 'monthly_cost' in column:
			good_features.append(column)
		if 'date' in column:
			good_features.append(column)
		if 'comp_num_sqft' in column:
			good_features.append(column)
		if 'comp_price' in column:
			good_features.append(column)
		# if 'building_num_units' in column:
		# 	good_features.append(column)
	# return df[good_features]
	return df.drop(labels=['price'], axis=1)


def get_y_dataframe(df):
	# return np.log(df.price)
	return df.price / df.price.mean()


def drop_const_columns(df_train, df_test):
	for column in df_train:
		if df_train[column].unique().size == 1:
			df_train.drop(column, axis=1, inplace=True)
			df_test.drop(column, axis=1, inplace=True)


def evaluate_model(model, X_train, X_test, y_train, y_test):
	yhat_train = model.predict(X_train)
	yhat_test = model.predict(X_test)

	print mean_squared_error(yhat_train, y_train)
	print mean_squared_error(yhat_test, y_test)
	print mean_absolute_error(yhat_train, y_train)
	print mean_absolute_error(yhat_test, y_test)
	print median_absolute_error(yhat_train, y_train)
	print median_absolute_error(yhat_test, y_test)
	print r2_score(yhat_train, y_train)
	print r2_score(yhat_test, y_test)


def build_elastic_net_model_for_catetory(df_train, df_test, category_name, alpha, l1_ratio):
	df_train_subset = df_train[df_train[category_name] == 1].copy()
	df_test_subset = df_test[df_test[category_name] == 1].copy()

	drop_const_columns(df_train_subset, df_test_subset)

	y_train = get_y_dataframe(df_train_subset)
	X_train = get_X_dataframe(df_train_subset)
	y_test = get_y_dataframe(df_test_subset)
	X_test = get_X_dataframe(df_test_subset)

	scaler = preprocessing.StandardScaler().fit(X_train)
	X_scaled_train = pd.DataFrame(scaler.transform(X_train))
	X_scaled_train.columns = X_train.columns
	X_scaled_train.index = X_train.index
	X_scaled_test = pd.DataFrame(scaler.transform(X_test))
	X_scaled_test.columns = X_test.columns
	X_scaled_test.index = X_test.index

	model = ElasticNet(l1_ratio=l1_ratio, alpha=alpha).fit(X_scaled_train, y_train)
	evaluate_model(model, X_scaled_train, X_scaled_test, y_train, y_test)

	return model


if __name__ == '__main__':

	data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
	df_train = pd.read_csv(os.path.join(data_dir, 'features.train.csv.gz'), header=0, index_col=0)
	df_test = pd.read_csv(os.path.join(data_dir, 'features.test.csv.gz'), header=0, index_col=0)

	thresh_lo = df_train.price.quantile(0.05)
	thresh_hi = df_train.price.quantile(0.95)
	df_train = df_train[(df_train.price > thresh_lo) & (df_train.price < thresh_hi)]  # .sample(n=100000)
	df_test = df_test[(df_test.price > thresh_lo) & (df_test.price < thresh_hi)]

	drop_const_columns(df_train, df_test)

	X_train = get_X_dataframe(df_train)
	y_train = get_y_dataframe(df_train)
	X_test = get_X_dataframe(df_test)
	y_test = get_y_dataframe(df_test)

	scaler = preprocessing.StandardScaler().fit(X_train)
	X_scaled_train = pd.DataFrame(scaler.transform(X_train))
	X_scaled_train.columns = X_train.columns
	X_scaled_train.index = X_train.index
	X_scaled_test = pd.DataFrame(scaler.transform(X_test))
	X_scaled_test.columns = X_test.columns
	X_scaled_test.index = X_test.index


	# cse cross validation to determine optimal hyperparameters for elastic net model
	enet_models = []
	for l1_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
		print '------- running cross-validation (l1_ratio = %f) -------' % l1_ratio
		enet_models.append(ElasticNetCV(cv=5, l1_ratio=l1_ratio, max_iter=10000, alphas=[2 ** i for i in range(-13, -5)]).fit(X_scaled_train, y_train))

	best_model = None
	for enet_model in enet_models:
		if best_model == None or np.min(enet_model.mse_path_) < np.min(best_model.mse_path_):
			best_model = enet_model

	evaluate_model(best_model, X_scaled_train, X_scaled_test, y_train, y_test)

	# plot CV curve of best model
	plt.figure()
	plt.plot(-np.log10(best_model.alphas_), best_model.mse_path_.mean(axis=-1), 'k', label='cv score')
	plt.axvline(-np.log10(best_model.alpha_), linestyle='--', color='k', label='best alpha')
	plt.legend()
	plt.xlabel('-log(alpha)')
	plt.ylabel('Mean Square Error')
	plt.axis('tight')
	plt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'modeling/elastic-net-cv.png')
	plt.savefig(plt_path)


	submodels = dict()
	MIN_TRAIN_SIZE_COUNT = 500
	for column in df_train.columns:
		if column.startswith('community_district_') or column.startswith('borough_'):
			train_size = np.sum(df_train[column])
			test_size = np.sum(df_test[column])
			if train_size >= MIN_TRAIN_SIZE_COUNT:
				print '------- building model for %s -------' % column
				print 'train set size: %d' % train_size
				print 'test  set size: %d' % test_size
				submodels[column] = build_elastic_net_model_for_catetory(df_train, df_test, category_name=column, alpha=best_model.alpha_, l1_ratio=best_model.l1_ratio)
			else:
				print  '------- not building model for %s (train size = %d) -------' % (column, train_size)
