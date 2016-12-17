import gzip
import pandas as pd
import numpy as np

test_saleids = pd.read_csv('test_saleids.txt', index_col=None, header=None)
train_saleids = pd.read_csv('train_saleids.txt', index_col=None, header=None)

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

df.reset_index(inplace=True)
df.columns = ['saleid','price','description']
df.price = np.log10(df.price)

all_train_data = df.loc[df['saleid'].isin(train_saleids[0].values),:]
all_test_data = df.loc[df['saleid'].isin(test_saleids[0].values),:]

all_train_data.to_csv('all_train_data.csv', index=True)
all_test_data.to_csv('all_test_data.csv', index=True)
