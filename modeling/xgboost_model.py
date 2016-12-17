from modeling_utils import evaluate_model, get_X_dataframe, get_y_dataframe, drop_const_columns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import pickle
import re


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = []

    def write(self, message):
        self.terminal.write(message)
        self.log.append(message)

    def flush(self):
        pass

    def reset_log(self):
        self.log = []


if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            num_boost_round = 3  # For final model
            n_estimators = 2  # For max_depth grid search
            max_depths = [2, 3, 4, 5]
    else:
        num_boost_round = 1000  # For final model
        n_estimators = 100
        max_depths = [4, 6, 8, 10]


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
    #------------------SCALE/SPLIT DATA--------------------#
    #------------------------------------------------------#

    print "Scaling data..."

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_scaled_train = scaler.transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    print "Making train/val splits..."

    X_train_final, X_val, y_train_final, y_val = train_test_split(X_scaled_train, y_train, test_size=0.2, random_state=11352)


    DMatrix_train = xgb.DMatrix(X_train_final,
                                label=y_train_final,
                                feature_names=X_train.columns)
    DMatrix_eval = xgb.DMatrix(X_val,
                               label=y_val,
                               feature_names=X_train.columns)

    #------------------------------------------------------#
    #------RUN GRIDSEARCH TO LEARN OPTIMAL DEPTH-----------#
    #------------------------------------------------------#

    # Start logging to save verbose_eval...
    stdout = sys.stdout
    sys.stdout = Logger()

    lowest_val_mse = None  # Initialize

    for depth in max_depths:
        sys.stdout.reset_log()

        bst = xgb.train({'max_depth':depth},
                        DMatrix_train,
                        n_estimators,
                        [(DMatrix_train, 'Train'), (DMatrix_eval, 'Validation')],
                        verbose_eval=True,
                        early_stopping_rounds=10,
                        )

        # Process results and save in DF
        eval_results = sys.stdout.log
        eval_results = [x.strip() for x in eval_results if x.find('rmse:') > 0]
        eval_results = [re.findall("\d+\.\d+", x) for x in eval_results]
        # Square to remove root from RMSE- map RMSE -> MSE
        train_results = [float(x[0]) ** 2 for x in eval_results]
        val_results = [float(x[1]) ** 2 for x in eval_results]
        num_rounds_boosting = range(1, len(eval_results) + 1)

        lc_df = {'max_depth':np.repeat(depth, len(num_rounds_boosting)),
                 'num_rounds_boosting':num_rounds_boosting,
                 'train_mse':train_results,
                 'val_mse':val_results}
        lc_df = pd.DataFrame.from_dict(lc_df)
        if lowest_val_mse is None:
            depth_all_results = lc_df
            best_depth = 1
            lowest_val_mse = min(val_results)
        else:
            depth_all_results = depth_all_results.append(lc_df, ignore_index=True)
            min_val = min(val_results)
            if min_val < lowest_val_mse:
                best_depth = depth
                lowest_val_mse = min_val
    # reset sys.stdout
    sys.stdout = stdout

    #------------------------------------------------------#
    #--------------MAKE PLOTS AND SAVE STATS---------------#
    #------------------------------------------------------#

    model_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_output/xgb1')
    depth_all_results.to_csv(os.path.join(model_output_dir, 'XGB_depth_learning_curve.csv'), index=False)
    colors = ['b', 'g', 'r', 'k']
    colors = dict(zip(max_depths, colors))

    # Make learning curve
    fig = plt.figure()
    ax = plt.subplot(111)
    for depth in max_depths:
        mask = (depth_all_results['max_depth'] == depth)
        ax.plot(depth_all_results.loc[mask, 'num_rounds_boosting'],
                 depth_all_results.loc[mask, 'train_mse'],
                 label="Train, Depth = {}".format(depth),
                 linestyle='dashed',
                 color=colors[depth])
        ax.plot(depth_all_results.loc[mask, 'num_rounds_boosting'],
                 depth_all_results.loc[mask, 'val_mse'],
                 label="Val, Depth = {}".format(depth),
                 linestyle='solid',
                 color=colors[depth])
    plt.title('Learning tree depth')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Number of Rounds of Boosting')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    fig.savefig(os.path.join(model_output_dir, 'XGB_depth_learning_lc.png'))


    #------------------------------------------------------#
    #----RUN GRIDSEARCH TO LEARN NUM ROUNDS AND ETA--------#
    #------------------------------------------------------#

    print 'Now learning number of rounds of boosting and learning rate using optimal tree depth...'

    final_params = {'silent'    : 0,
                    'objective' : 'reg:linear',
                    'max_depth' : best_depth}

    # Start logging to save verbose_eval...
    stdout = sys.stdout
    sys.stdout = Logger()

    etas = [0.02, 0.1, 0.3, 0.7]
    lowest_val_mse = None  # Initialize

    for eta in etas:
        sys.stdout.reset_log()
        final_params['learning_rate'] = eta

        bst = xgb.train(final_params,
                        DMatrix_train,
                        num_boost_round,
                        [(DMatrix_train, 'Train'), (DMatrix_eval, 'Validation')],
                        verbose_eval=True,
                        early_stopping_rounds=10,
                        )

        # Process results and save in DF
        eval_results = sys.stdout.log
        eval_results = [x.strip() for x in eval_results if x.find('rmse:') > 0]
        eval_results = [re.findall("\d+\.\d+", x) for x in eval_results]
        # Square to remove root from RMSE- map RMSE -> MSE
        train_results = [float(x[0]) ** 2 for x in eval_results]
        val_results = [float(x[1]) ** 2 for x in eval_results]
        num_rounds_boosting = range(1, len(eval_results) + 1)

        lc_df = {'eta':np.repeat(eta, len(num_rounds_boosting)),
                 'num_rounds_boosting':num_rounds_boosting,
                 'train_mse':train_results,
                 'val_mse':val_results}
        lc_df = pd.DataFrame.from_dict(lc_df)
        if lowest_val_mse is None:
            all_results = lc_df
            best_eta = eta
            best_etas_num_rounds = bst.best_ntree_limit
            lowest_val_mse = min(val_results)
        else:
            all_results = all_results.append(lc_df, ignore_index=True)
            min_val = min(val_results)
            if min_val < lowest_val_mse:
                best_eta = eta
                lowest_val_mse = min_val
                best_etas_num_rounds = bst.best_ntree_limit
    # reset sys.stdout
    sys.stdout = stdout

    #------------------------------------------------------#
    #--------------MAKE PLOTS AND SAVE STATS---------------#
    #------------------------------------------------------#

    all_results.to_csv(os.path.join(model_output_dir, 'XGB_learning_curve.csv'), index=False)
    colors = ['b', 'g', 'r', 'k']
    colors = dict(zip(etas, colors))
    # Make learning curve
    fig = plt.figure()
    ax = plt.subplot(111)
    for eta in etas:
        mask = (all_results['eta'] == eta)
        ax.plot(all_results.loc[mask, 'num_rounds_boosting'],
                 all_results.loc[mask, 'train_mse'],
                 label="Train, eta = {}".format(eta),
                 linestyle='dashed',
                 color=colors[eta])
        ax.plot(all_results.loc[mask, 'num_rounds_boosting'],
                 all_results.loc[mask, 'val_mse'],
                 label="Val, eta = {}".format(eta),
                 linestyle='solid',
                 color=colors[eta])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    plt.title('Learning number of rounds and learning rate')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Number of Rounds of Boosting')
    fig.savefig(os.path.join(model_output_dir, 'XGB_num_round_eta_lc.png'))

    #------------------------------------------------------#
    #------LEARN FINAL MODEL OVER ALL TRAINING DATA--------#
    #------------------------------------------------------#

    print 'Now learning final model...'
    # Add additional learned parameters:
    final_params['learning_rate'] = best_eta

    DMatrix_train_all = xgb.DMatrix(X_scaled_train,
                                label=y_train,
                                feature_names=X_train.columns)

    DMatrix_test = xgb.DMatrix(X_scaled_test,
                               label=y_test,
                               feature_names=X_test.columns)

    bst_final = xgb.train(final_params,
                    DMatrix_train_all,
                    best_etas_num_rounds,
                    [(DMatrix_train, 'Train')],
                    verbose_eval=True,
                    )

    print 'Making importance plot for final model...'

    importance = bst_final.get_score(importance_type='gain')
    importance = sorted(importance.items(), key=(lambda x: x[1]), reverse=True)

    # Now normalize
    importance = pd.DataFrame(importance, columns=['feature', 'fscore'])
    importance['fscore'] = importance['fscore'] / importance['fscore'].max()

    # Subset top 10 features
    importance = importance.loc[range(9, -1, -1), :]

    # Now make and save fig.
    importance.plot(kind='barh', x='feature', y='fscore', xlim=(0, 1), legend=False)
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Relative importance (normalized gain score)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, 'XGB_feature_importance.png'))

    #------------------------------------------------------#
    #--------------GET TEST SET RESULTS AND SAVE-----------#
    #------------------------------------------------------#

    print "Evaluating model..."

    predicted_test = bst_final.predict(DMatrix_test, ntree_limit=bst_final.best_ntree_limit)
    evaluate_model('XGB', predicted_test, y_test, model_output_dir)

    print "Saving model..."

    bst_final.save_model(os.path.join(model_output_dir, 'XGB_final_model.model'))
