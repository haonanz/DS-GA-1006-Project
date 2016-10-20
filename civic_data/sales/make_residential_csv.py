import pandas as pd

years = range(2011,2016)
boros = ['queens', 'bronx', 'statenisland', 'manhattan', 'brooklyn']

dfs = []
for year in years:
    for boro in boros:
        dfs.append(pd.read_excel('{}_{}.xls'.format(year, boro), header=4))

for df in dfs:
    df.columns = pd.Index([n.strip('\n') for n in df.columns])

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.append(df)

# Note tax class 1 and 2 includes residential properties
# https://www1.nyc.gov/site/finance/taxes/definitions-of-property-assessment-terms.page
residential = merged.loc[merged['TAX CLASS AT TIME OF SALE'].isin([1,2,'1','2', '2A','2C','1A','1B','2B','1C','1D'])]

residential.to_csv('all_residential_2011_2015.csv', index=False)
