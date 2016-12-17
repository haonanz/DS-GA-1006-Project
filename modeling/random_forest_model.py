from modeling_utils import evaluate_model, get_X_dataframe, get_y_dataframe, drop_const_columns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys


if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            params = {'n_estimators':[2, 4],
                      'max_depth':[4, 8],
                      'criterion':['mse'],
                      'oob_score':[True],
                      'n_jobs':[-1],
                      'verbose':[2],
                      'max_features':['auto', 'sqrt']
                      }
    else:
        params = {'n_estimators':[50, 100, 200, 400],
                  'max_depth':[4, 8],
                  'criterion':['mse'],
                  'oob_score':[True],
                  'n_jobs':[-1],
                  'verbose':[2],
                  'max_features':['auto', 'sqrt']
                  }

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

    df_train = df_train[(df_train.price > thresh_lo) & (df_train.price < thresh_hi)]  # .sample(n=100000)
    df_test = df_test[(df_test.price > thresh_lo) & (df_test.price < thresh_hi)]
    drop_const_columns(df_train, df_test)

    X_train = get_X_dataframe(df_train)
    y_train = get_y_dataframe(df_train)
    X_test = get_X_dataframe(df_test)
    y_test = get_y_dataframe(df_test)

    #------------------------------------------------------#
    #--------------------SCALE DATA------------------------#
    #------------------------------------------------------#

    print "Scaling data..."

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)


    #------------------------------------------------------#
    #---------RUN GRIDSEARCH TO LEARN OPTIMAL RF-----------#
    #------------------------------------------------------#

    print 'Fitting random forest...'

    first_model = True
    best_score = 0  # Initialize best_score- note we'll reset for first model
    all_scores = {'n_estimators':[],
                  'max_depth':[],
                  'max_features':[],
                  'mse':[]}
    for g in ParameterGrid(params):
        print "\tTesting {}...".format(g)
        RF = RandomForestRegressor(**g)
        RF.fit(X_scaled_train, y_train)
        mse = mean_squared_error(y_train, RF.oob_prediction_)
        print "\OOB MSE: %0.5f\n" % mse
        # Save results to create CF results df and plot learning curve
        all_scores['n_estimators'].append(g['n_estimators'])
        all_scores['max_depth'].append(g['max_depth'])
        all_scores['max_features'].append(g['max_features'])
        all_scores['mse'].append(mse)

        if first_model or mse < best_score:
            first_model = False
            best_score = mse
            best_grid = g
            best_rf = RF

    print 'Final model:'
    print "\tOOB MSE: %0.5f" % best_score
    print "\tGrid:", best_grid


    #------------------------------------------------------#
    #--------------MAKE PLOTS AND SAVE STATS---------------#
    #------------------------------------------------------#

    print 'Making learning curves and saving summary stats...'

    model_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_output/rf')
    results_df = pd.DataFrame.from_dict(all_scores)

    fig = plt.figure()
    ax = plt.subplot(111)
    for depth in params['max_depth']:
        for features in params['max_features']:
            mask = ((results_df['max_features'] == features) & (results_df['max_depth'] == depth))
            ax.plot(results_df.loc[mask, 'n_estimators'],
                     results_df.loc[mask, 'mse'],
                     label='{}, {}'.format(features, depth))
    plt.title('OOB Optimization of RF max_features, max_depth, n_estimators')
    plt.xlabel('n_estimators (Number of Trees in Forest)')
    plt.ylabel('OOB Mean Squared Error')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left',
              bbox_to_anchor=(1, 0.5),
              frameon=True,
              title='Max: Features, Depth')

    plt.savefig(os.path.join(model_output_dir, 'RF_cv_learning_curve.png'))

    results_df.to_csv(os.path.join(model_output_dir, 'RF_cv_results.csv'), index=False)


    print 'Making importance plot...'

    # Now make importance plot using RF

    importance = RF.feature_importances_
    assert len(importance) == len(X_train.columns), 'fscores length not same as X_train.columns'
    importance = zip(list(X_train.columns), list(importance))

    importance = sorted(importance, key=(lambda x: x[1]), reverse=True)
    # Now normalize
    importance = pd.DataFrame(importance, columns=['feature', 'fscore'])
    importance['fscore'] = importance['fscore'] / importance['fscore'].max()

    # Subset top 10 features
    importance = importance.loc[range(9, -1, -1), :]

    # Now make and save fig.
    importance.plot(kind='barh', x='feature', y='fscore', xlim=(0, 1), legend=False)
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Feature')
    plt.xlabel('Relative importance (normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'RF_feature_importance.png'))


    #------------------------------------------------------#
    #--------------MAKE PREDICTIONS AND SAVE---------------#
    #------------------------------------------------------#

    print "Evaluating model..."

    predicted_test = RF.predict(X_scaled_test)

    evaluate_model('RF', predicted_test, y_test, model_output_dir)

    print "Saving model..."

    with open(os.path.join(model_output_dir, 'RF_final_model.model'), 'wb') as outfile:
        pickle.dump(RF, outfile)
