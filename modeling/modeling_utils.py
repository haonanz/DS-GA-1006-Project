from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


def percentage_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100.0


def mean_percentage_error(y_true, y_pred):
    return np.mean(percentage_error(y_true, y_pred))


def median_percentage_error(y_true, y_pred):
    return np.median(percentage_error(y_true, y_pred))


def get_X_dataframe(df):
    return df.drop(labels=['price'], axis=1)


def get_y_dataframe(df):
    return df.price / df.price.mean()


def drop_const_columns(X_train, X_test):
    for column in X_train:
        if X_train[column].unique().size == 1:
            X_train.drop(column, axis=1, inplace=True)
            X_test.drop(column, axis=1, inplace=True)


def evaluate_model(model_name, predicted_test, y_test, output_dir):
    metrics = [r2_score, mean_squared_error, mean_absolute_error, mean_percentage_error, median_absolute_error, median_percentage_error]
    with open(os.path.join(output_dir, '%s_test_scores.txt' % model_name), 'w') as test_output:
        test_output.write('Test set scores:\n')
        for metric in metrics:
            metric_score = metric(y_test, predicted_test)
            test_output.write("{} = {}\n".format(metric.__name__, str(metric_score)))

    perc_error = percentage_error(y_test, predicted_test)
    median_perc_error = np.median(perc_error)
    num_predicted = float(len(y_test))
    within_5 = len(perc_error[perc_error <= 5.0]) / num_predicted * 100.0
    within_10 = len(perc_error[perc_error <= 10.0]) / num_predicted * 100.0
    within_20 = len(perc_error[perc_error <= 20.0]) / num_predicted * 100.0

    fig = plt.figure()
    plt.hist(perc_error[perc_error <= 200.0],
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
    plt.xlim(0, 200)
    plt.savefig(os.path.join(output_dir, '%s_perc_error_hist.png' % model_name))
    plt.close(fig)

    pd.DataFrame(predicted_test, index=y_test.index).to_csv(os.path.join(output_dir, '%s_test_predictions.csv' % model_name), index=True)