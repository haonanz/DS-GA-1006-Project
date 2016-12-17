import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------------------------#

#Read in  data
print 'Reading in data...'
#Note the shapes of train_X and val_X are transposed.

train_X = pd.read_csv('sLDA_10_train.csv',index_col=None, header=None)
train_X = train_X.values.T
val_X = pd.read_csv('sLDA_10_val.csv',index_col=None, header=None)
val_X  = val_X.values.T

train_price = pd.read_csv('sLDA_10_train_price.csv',index_col=None, header=None)
train_price = np.ravel(train_price)
val_price = pd.read_csv('sLDA_10_val_price.csv',index_col=None, header=None)
val_price = np.ravel(val_price)


print 'Learning GBR model in Python using sklearn...'
gbr = GradientBoostingRegressor()
gbr.fit(train_X, train_price)

print "\tGetting test-set GBR score..."
#Transform test_docs to topic weight matrix
test_pred = gbr.predict(val_X)
mse = mean_squared_error(val_price, test_pred)

print 'MSE using GBR on k=10 from sLDA = {}'.format(mse)

print "Learning baseline model..."
#Baseline model: just predict the average log price
baseline_pred = np.ones_like(val_price)*np.mean(train_price)
baseline_mse = mean_squared_error(val_price, baseline_pred)

print 'Making histogram of residuals for K = 5 model...'
plt.figure(figsize=(8,6))
plt.hist(val_price - baseline_pred, bins=100, label='Baseline: K = 0 (predict mean)', alpha=0.5)
plt.hist(val_price -  test_pred,
             bins=100, label='sLDA, K = 10 using Gradient Boosting', alpha=0.5)
plt.legend()
plt.ylabel('Frequency')
plt.title('Comparing the distribution of residuals by K')
plt.xlabel('Residual (log(price) - predicted(log(price)))')
plt.legend(loc='best')
plt.savefig('figs/final_residuals.png')


