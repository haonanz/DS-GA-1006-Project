from modeling_utils import evaluate_model, get_X_dataframe, get_y_dataframe, drop_const_columns
from sklearn import preprocessing
from sklearn.linear_model import ElasticNetCV

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


CV_ALPHAS = [2 ** i for i in reversed(range(-15, -4))]
L1_RATIO_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]


def build_elastic_net_model(df_train, df_test, output_dir, model_name):
	drop_const_columns(df_train, df_test)
	X_train = get_X_dataframe(df_train)
	y_train = get_y_dataframe(df_train)
	X_test = get_X_dataframe(df_test)
	y_test = get_y_dataframe(df_test)

	scaler = preprocessing.StandardScaler().fit(X_train)
	X_scaled_train = scaler.transform(X_train)
	X_scaled_test = scaler.transform(X_test)

	enet_models = []
	for l1_ratio in L1_RATIO_GRID:
		print 'running cross validation with l1_ratio = %f for %s' % (l1_ratio, model_name)
		enet_models.append(ElasticNetCV(
			cv=3,
			l1_ratio=l1_ratio,
			alphas=CV_ALPHAS,
			selection='random',
			random_state=11352,
			max_iter=10000
			).fit(X_scaled_train, y_train))

	best_model = None
	for enet_model in enet_models:
		if best_model == None or np.min(enet_model.mse_path_) < np.min(best_model.mse_path_):
			best_model = enet_model

	predicted_test = best_model.predict(X_scaled_test)
	evaluate_model(model_name, predicted_test, y_test, output_dir=output_dir)

	with open(os.path.join(output_dir, '%s_final_model.model' % model_name), 'wb') as outfile:
		pickle.dump(best_model, outfile)

	return best_model, enet_models


if __name__ == '__main__':

	#------------------------------------------------------#
	#------------------READ IN DATA------------------------#
	#------------------------------------------------------#

	print "Reading in data..."

	data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
	df_train = pd.read_csv(os.path.join(data_dir, 'features.train.csv.gz'), header=0, index_col=0)
	df_test = pd.read_csv(os.path.join(data_dir, 'features.test.csv.gz'), header=0, index_col=0)

	#------------------------------------------------------#
	#-------------------FILTER DATA------------------------#
	#------------------------------------------------------#

	print "Filtering data..."

	thresh_lo = df_train.price.quantile(0.05)
	thresh_hi = df_train.price.quantile(0.95)

	df_train = df_train[(df_train.price > thresh_lo) & (df_train.price < thresh_hi)]  # .sample(n=40000)
	df_test = df_test[(df_test.price > thresh_lo) & (df_test.price < thresh_hi)]  # .sample(n=10000)

	#------------------------------------------------------#
	#-------------------BUILD MODEL------------------------#
	#------------------------------------------------------#

	model_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_output/enet')

	best_model, enet_models = build_elastic_net_model(
		df_train,
		df_test,
		output_dir=model_output_dir,
		model_name='enet')

	print 'best model: l1_ratio = %f, alpha = %f' % (best_model.l1_ratio, best_model.alpha_)

	#------------------------------------------------------#
	#--------------MAKE PLOTS AND SAVE STATS---------------#
	#------------------------------------------------------#

	fig = plt.figure()
	ax = plt.subplot()

	for idx in range(len(L1_RATIO_GRID)):
		model = enet_models[idx]
		l1_ratio = L1_RATIO_GRID[idx]
		plt.plot(-np.log10(model.alphas_), model.mse_path_.mean(axis=-1), label='l1_ratio = %.1f' % l1_ratio)

	plt.title('Elastic Net Model Cross Validation')
	plt.xlabel('-log(alpha)')
	plt.ylabel('Average Cross Validation MSE')
	plt.legend()

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
	plt.savefig(os.path.join(model_output_dir, 'elastic-net-cv.png'))

	#------------------------------------------------------#
	#--------------BUILD MODEL FOR SUBREGIONS--------------#
	#------------------------------------------------------#

	submodel_output_dir = os.path.join(model_output_dir, 'submodels')
	submodels = dict()
	MIN_TRAIN_SIZE_COUNT = 1000
	for column in df_train.columns:
		if column.startswith('borough_') or column.startswith('community_district_') or \
				(column.startswith('neighborhood_') and not column.startswith('neighborhood_comp_')):
			train_size = np.sum(df_train[column])
			test_size = np.sum(df_test[column])
			if train_size >= MIN_TRAIN_SIZE_COUNT:
				print '------- building model for %s -------' % column
				print 'train set size: %d' % train_size
				print 'test  set size: %d' % test_size
				df_train_subset = df_train[df_train[column] == 1].copy()
				df_test_subset = df_test[df_test[column] == 1].copy()
				build_elastic_net_model(
					df_train_subset,
					df_test_subset,
					output_dir=submodel_output_dir,
					model_name='enet_%s' % column)
			else:
				print  '------- not building model for %s (train size = %d) -------' % (column, train_size)
