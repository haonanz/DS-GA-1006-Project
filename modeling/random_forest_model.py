from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from linearmodel import get_X_dataframe, get_y_dataframe, drop_const_columns, evaluate_model
import pickle
import sys

def percentage_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100.0

if __name__ == '__main__':

    if len(sys.argv) > 0:
        if sys.argv[1] == 'test':
            params = {'n_estimators':[2, 4],
                      'max_depth':[4,8],
                      'criterion':['mse'],
                      'oob_score':[True],
                      'n_jobs':[-1],
                      'verbose':[2],
                      'max_features':['auto', 'sqrt']
                      }
        else:
            params = {'n_estimators':[50, 100, 200, 400],
                      'max_depth':[4,8],
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
    best_score = 0 #Initialize best_score- note we'll reset for first model
    all_scores = {'n_estimators':[],
                  'max_depth':[],
                  'max_features':[],
                  'mse':[]}
    for g in ParameterGrid(params):
        print "\tTesting {}...".format(g)
        RF = RandomForestRegressor(**g)
        RF.fit(X_scaled_train,y_train)
        mse = mean_squared_error(y_train, RF.oob_prediction_)
        print "\OOB MSE: %0.5f\n" % mse
        #Save results to create CF results df and plot learning curve
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

    results_df = pd.DataFrame.from_dict(all_scores)

    fig = plt.figure()
    ax = plt.subplot(111)
    for depth in params['max_depth']:
        for features in params['max_features']:
            mask = ((results_df['max_features']==features) & (results_df['max_depth']==depth))
            ax.plot(results_df.loc[mask,'n_estimators'],
                     results_df.loc[mask,'mse'],
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

    plt.savefig('./figs/RF_cv_learning_curve.png')

    results_df.to_csv('./saved_models/RF_cv_results.csv', index=False)


    print 'Making importance plot...'

    #Now make importance plot using RF

    importance = RF.feature_importances_
    assert len(importance) == len(X_train.columns), 'fscores length not same as X_train.columns'
    importance = zip(list(X_train.columns), list(importance))

    importance = sorted(importance, key=(lambda x: x[1]), reverse=True)
    #Now normalize
    importance = pd.DataFrame(importance, columns=['feature', 'fscore'])
    importance['fscore'] = importance['fscore'] / importance['fscore'].max()

    #Subset top 10 features
    importance = importance.loc[range(9,-1,-1),:]

    #Now make and save fig.
    importance.plot(kind='barh', x='feature', y='fscore', xlim=(0,1), legend=False)
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Feature')
    plt.xlabel('Relative importance (normalized)')
    plt.tight_layout()
    plt.savefig('./figs/RF_feature_importance.png')


    #------------------------------------------------------#
    #--------------MAKE PREDICTIONS AND SAVE---------------#
    #------------------------------------------------------#
    print "Saving RF model..."

    predicted_test = RF.predict(X_scaled_test)
    metrics = [mean_squared_error, r2_score, mean_absolute_error, median_absolute_error]
    with file('./saved_models/RF_test_scores.txt','w') as test_output:
        test_output.write('Test set scores:\n')
        for metric in metrics:
            metric_score = metric(y_test, predicted_test)
            test_output.write("{} = {}\n".format(metric.__name__, str(metric_score)))

    perc_error = percentage_error(y_test, predicted_test)
    median_perc_error = np.median(perc_error)
    num_predicted = float(len(y_test))
    within_5 = len(perc_error[perc_error<=5.0])/num_predicted * 100.0
    within_10 = len(perc_error[perc_error<=10.0])/num_predicted * 100.0
    within_20 = len(perc_error[perc_error<=20.0])/num_predicted * 100.0
    plt.figure()
    plt.hist(perc_error[perc_error<=200.0],
             bins=40)
    plt.axvline(5,
                linestyle='--',
                color='g',
                label='Within 5%: {0:.1f}%'.format(within_5))
    plt.axvline(10,
                linestyle='--',
                color='y',
                label='Within 10%: {0:.1f}%'.format(within_10))
    plt.axvline(20,
                linestyle='--',
                color='r',
                label='Within 20%: {0:.1f}%'.format(within_20))
    plt.axvline(median_perc_error,
                linestyle='--',
                color='k',
                label='Median: {0:.1f}%'.format(median_perc_error))
    plt.title('Absolute Percentage Error on Predicted Price')
    plt.xlabel('Absolute percentage error')
    plt.ylabel('Frequency (counts)')
    plt.legend()
    plt.xlim(0,200)
    plt.savefig('./figs/RF_perc_error_hist.png')

    pd.DataFrame(predicted_test, index=y_test.index).to_csv('./saved_models/RF_test_predictions.csv', index=True)

    with open('./saved_models/RF_final_model.model','wb') as outfile:
        pickle.dump(RF, outfile)
