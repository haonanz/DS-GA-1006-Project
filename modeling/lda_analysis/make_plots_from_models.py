import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

#Use for saving models
import pickle

#----------------------------------------------------------------#


with open("LDA_models.p", "rb") as f:
    models = pickle.load(f)

sLDA_MSEs = pd.read_csv('sLDA_MSEs.csv', header=None, index_col=False)

#Now make histogram of residuals for best python model

print 'Making histogram of residuals for K = 5 model...'
plt.figure(figsize=(8,6))
plt.hist(Test_price - baseline_pred, bins=100, label='Baseline: K = 0 (predict mean)', alpha=0.5)
plt.hist(Test_price -  models[5]['test_pred_GBR'],
             bins=100, label='K = {} using Gradient Boosting'.format(5), alpha=0.5)
plt.legend()
plt.ylabel('Frequency')
plt.title('Comparing the distribution of residuals by K')
plt.xlabel('Residual (log(price) - predicted(log(price)))')
plt.legend(loc='best')
plt.savefig('figs/residuals.png')

#Now make learning curve including R

print 'Making learning curves for three models...'
plt.figure(figsize=(8,6))
plt.plot([0] + models.keys(), [baseline_mse] + [models[K]['mse_GBR'] for K in models], label='Gradient Boosted model for LDA weights')
plt.plot([0] + models.keys(), [baseline_mse] + [models[K]['mse_LS'] for K in models], label='Linear model for LDA weights')
plt.plot([0] + list(sLDA_MSEs.loc[:,0]), [baseline_mse] + list(sLDA_MSEs.loc[:,1]), label='Linear model for sLDA weights')
plt.xlabel('Number of topics K')
plt.ylabel('MSE for log(price) prediction')
plt.title('Learning curve: Optimizing K')
plt.legend(loc='best')
plt.savefig('figs/learning_curve.png')

