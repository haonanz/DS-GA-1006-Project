import pandas as pd

boro_ids = ['BK','BX','MN','QN','SI']

features_to_keep = ['Borough',
                    'BBL',
                    'Block',
                    'Lot',
                    'Address',
                    'SchoolDist',
                    'PolicePrct',
                    'CT2010',
                    'ZipCode',
                    'LotArea',
                    'BldgArea',
                    'ResArea',
                    'CondoNo',
                    'NumBldgs',
                    'NumFloors',
                    'YearBuilt',
                    'YearAlter1',
                    'YearAlter2',
                    'AssessLand',
                    'AssessTot',]

dfs = []
for i in boro_ids:
    dfs.append(pd.read_csv('{}.csv'.format(i), usecols=features_to_keep))

merged = dfs[0]

for i in range(1,len(dfs)):
    merged = merged.append(dfs[i])

merged.to_csv('all_pluto.csv', index=False)
