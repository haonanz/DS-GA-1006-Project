import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from linearmodel import get_X_dataframe, get_y_dataframe, drop_const_columns, evaluate_model

#------------------------------------------------------#
#------------------READ IN DATA------------------------#
#------------------------------------------------------#
print "Reading in data..."
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
yhat = pd.read_csv('./saved_models/XGB_model_3_test_predictions.csv', index_col=0)
yhat.columns = ['pred']

print yhat.head

#Note have to read in all data since thresholding done in modeling scripts

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
y = df_test['price']
y = pd.DataFrame.from_dict({'price':y})
y.index = X_test.index

print "Joining data..."

df = y.join(yhat)

#------------------------------------------------------#
#------------------Get error arrays--------------------#
#------------------------------------------------------#

print "Getting errors..."

def abs_percentage_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100.0

def squared_error(y_true, y_pred):
    return (y_true - y_pred)**2

sqr_error = squared_error(df['price'],df['pred'])
pct_error = abs_percentage_error(df['price'],df['pred'])

print "Making plots.."

fig = plt.figure()
ax1 = plt.subplot(211)
ax1.scatter(df['price'], sqr_error, s=10, c='b', marker='o')
ax1.set_xlim([0, np.max(df['price'])+0.1])
ax1.set_ylim([0, np.max(sqr_error)+1])
ax1.set_ylabel('Squared error')
ax1.set_title('Normalized price vs. square error from XGB')

ax2 = plt.subplot(212)
ax2.scatter(df['price'], pct_error, s=10, c='g', marker='o')
ax2.set_title('Normalized price vs. absolute percent error from XGB')
ax2.set_ylabel('Absolute percent error')
ax2.set_xlabel('Normalized Price (price / mean price)')
ax2.set_xlim([0, np.max(df['price'])+0.1])
ax2.set_ylim([0, np.max(pct_error)+10])
plt.savefig('./figs/XGB_model_3_square_vs_pct_error.png')
