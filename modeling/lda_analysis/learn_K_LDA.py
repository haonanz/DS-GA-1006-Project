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

#Read in all data
print 'Reading in data...'
batches = range(1,5)
all_records = {}
num_records = 0
for batch in batches:
    with gzip.open('./../scripts/sales_listings.{}.csv.gz'.format(batch)) as f:
        lines = f.readlines()
    parsed = {}
    for line in lines[1:]:
        features = line.strip().split(',')
        saleid = features[0]
        price = float(features[2])
        description = features[-1]
        parsed[saleid] = {'price':price, 'description':description}

    num_records += len(parsed)
    all_records.update(parsed)

assert len(all_records) == num_records
df = pd.DataFrame.from_dict(all_records,orient='index')

print 'Filtering data:'
print '\tShape with null descriptions = ', df.shape
df = df.dropna(axis=0, how='any')
print '\tShape without null vals = ', df.shape
df = df.loc[df['price'] > 1000, :]
print '\tShape without null vals or <1000 price = ', df.shape

#Make global training test splits
#Save IDs for use in other scripts

train, test = train_test_split(df, test_size=0.2, random_state=42)

pd.Series(test.index).to_csv('test_saleids.txt', index=False)
pd.Series(train.index).to_csv('train_saleids.txt', index=False)

# Now make train/val split for optimizing K

Train_docs, Test_docs, Train_price, Test_price = train_test_split(train['description'], train['price'], test_size=0.2, random_state=42)

print 'Making price histograms...'

#Not log transformed
plt.figure(figsize=(8,6))
plt.hist(Train_price.values, bins=100)
plt.title('Histogram for untransformed prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('figs/untransformed_prices.png')

#log transformed
plt.figure(figsize=(8,6))
plt.hist(np.log10(Train_price.values), bins=100)
plt.title('Histogram for log transformed prices')
plt.xlabel('log(Price)')
plt.ylabel('Frequency')
plt.savefig('figs/log_transformed_prices.png')

#Use this log transform
Train_price = np.log10(Train_price.values)
Test_price = np.log10(Test_price.values)

print "Learning baseline model..."
#Baseline model: just predict the average log price
baseline_pred = np.ones_like(Test_price)*np.mean(Train_price)
baseline_mse = mean_squared_error(Test_price, baseline_pred)

print "Vectorizing training documents..."
#Note introduced hyperparameters: max_df, min_df
counter = CountVectorizer(stop_words='english', max_df = 0.9, min_df= 500)
train_counts = counter.fit_transform(Train_docs)

print "Writing files for R..."
#Write vocab to file for use in R (slda)
with open('vocab.txt', 'w') as vocab:
    for word in counter.vocabulary_.keys():
          vocab.write("%s\n" % word)
#Write train/test docs and counts to files for R
Train_docs.to_csv('train_docs.csv')
Test_docs.to_csv('test_docs.csv')
with open('train_price.csv', 'w') as vocab:
    for price in Train_price:
          vocab.write("%s\n" % price)
with open('test_price.csv', 'w') as vocab:
    for price in Test_price:
          vocab.write("%s\n" % price)

print 'Learning models in Python using sklearn...'
models = {}
num_topics_to_test = sorted(range(5,30,5) + [4,6,7,8,9])
for n_topics in num_topics_to_test:
    print "Running K = {}\n".format(n_topics)
    print "\tLearning LDA model..."
    #Learn LDA model and get topic weights for training docs
    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    max_iter=10,
                                    learning_method='online',
                                    n_jobs = -1, #Uses all cores,
                                    random_state=1234)
    topic_weights = lda.fit_transform(train_counts)

    #Learn GradientBoostedRegressor model of price ~ topic weights
    #Note this model was chosen since it is relatively immune to overfitting
    #and is non-linear (which is anticipated to improve performance)
    print "\tLearning GBR model..."
    gbr = GradientBoostingRegressor()
    gbr.fit(topic_weights, Train_price)

    print "\tGetting test-set GBR score..."
    #Transform test_docs to topic weight matrix
    test_topics = lda.transform(counter.transform(Test_docs))
    test_pred = gbr.predict(test_topics)
    mse = mean_squared_error(Test_price, test_pred)

    print "\tLearning linear model..."
    LS = LinearRegression(fit_intercept=False)
    LS.fit(topic_weights, Train_price)

    print '\tGetting test-set linear predictions...'
    linear_test_pred = LS.predict(test_topics)
    mse_LS = mean_squared_error(Test_price, linear_test_pred)

    #Save model and score:
    print "\tSaving model...\n"
    models[n_topics] = {'lda': lda,
                        'gbr': gbr,
                        'ls': LS,
                        'mse_GBR': mse,
                        'mse_LS': mse_LS,
                        'test_pred_GBR': test_pred,
                        'linear_test_pred': linear_test_pred}


print 'Pickling model...'
with open( "LDA_models.p", "wb" ) as fout:
    pickle.dump(models, fout)

#Now run R code to learn sLDA models

print 'Running R code for SLDA...'
robjects.r.source('sLDA.R') #Note this saves MSEs for sLDA in file sLDA_MSEs.csv
