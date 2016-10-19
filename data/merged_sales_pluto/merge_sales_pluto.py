import pandas as pd
import numpy as np


#First do naive merge on BBL:
print 'Merging all sale records with BBL matches in Pluto...'

pluto = pd.read_csv('./../pluto/all_pluto.csv')
sales = pd.read_csv('./../sales/all_residential_2011_2015.csv')

bbl_features = ['BOROUGH','BLOCK','LOT']

for feature in bbl_features:
    sales[feature] = sales[feature].astype(str)

def make_bbl(row):
    bbl = row['BOROUGH'] + row['BLOCK'].zfill(5) + row['LOT'].zfill(4)
    return bbl

sales['BBL'] = sales.apply(make_bbl, axis=1).astype(float)

merged = pd.merge(sales, pluto, how='left', on=['BBL'])
num_sales = merged.shape[0]

#Next use PAD to merge condos
print 'Reading in PAD to map condo BBLs to condo billing BBLs...'

PAD_cols = ['loboro', 'loblock','lolot', 'hiboro','hiblock','hilot','boro','block','lot','condoflag','billboro','billblock','billlot']
PAD = pd.read_csv('bobabbl.txt', index_col=None, usecols=PAD_cols)

PAD = PAD.loc[PAD['condoflag'] == 'C',:]
PAD = PAD.replace(r'\s+', np.nan, regex=True)
for col in PAD.columns:
    PAD[col] = PAD[col].astype(str)

del PAD['condoflag']
def make_BBL_range(row):
    all_condo = []
    lo_BBL = row['loboro'] + row['loblock'].zfill(5) + row['lolot'].zfill(4)
    hi_BBL = row['hiboro'] + row['hiblock'].zfill(5) + row['hilot'].zfill(4)
    if row['billboro'] == 'nan':
        bill_BBL = np.nan
    else:
        bill_BBL = row['billboro'] + row['billblock'].zfill(5) + row['billlot'].zfill(4)
        bill_BBL = int(bill_BBL)
    for bbl in range(int(lo_BBL),int(hi_BBL) + 1):
        all_condo.append((bbl, int(lo_BBL), bill_BBL))
    return all_condo

all_condo_bbls = []
for row in PAD.iterrows():
    all_condo_bbls.extend(make_BBL_range(row[1]))

condo_map = pd.DataFrame.from_records(all_condo_bbls)
condo_map.columns = ['unit_BBL','lo_BBL','bill_BBL']

unmatched = merged.loc[merged['Block'].isnull(),:].copy(deep=True)
unmatched = unmatched.dropna(axis=1)

merge_on_billing_BBL = pd.merge(unmatched, condo_map, how='left', left_on=['BBL'], right_on=['unit_BBL'])



print 'Merging unmatched sale records with pluto using condo billing BBLs...'

merged_with_pluto = pd.merge(merge_on_billing_BBL, pluto, how='left', left_on=['bill_BBL'], right_on=['BBL'])

merged_with_pluto.drop(['unit_BBL','lo_BBL','bill_BBL'], axis=1, inplace=True)
merged_with_pluto.rename(columns={'BBL_x': 'unit_BBL', 'BBL_y': 'Pluto_BBL'}, inplace=True)

merged['Pluto_BBL'] = merged['BBL']
merged.rename(columns={'BBL':'unit_BBL'}, inplace=True)

merged = merged.loc[~merged['Block'].isnull(),:].append(merged_with_pluto)
print 'Successsful merge of %.2f percent of the sales data with Pluto...' % ((1.0 -(num_sales - np.amin(merged.count().values))/ float(num_sales))*100)

print 'Saving final merged csv file...'

merged.to_csv('merged_sales_pluto_data.csv', index=False)
