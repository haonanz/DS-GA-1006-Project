#!/bin/bash

curl https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyc_pluto_16v1.zip > nyc_pluto_16v1.zip

unzip -a nyc_pluto_16v1.zip

rm nyc_pluto_16v1.zip

python make_pluto_csv.py

boroughs=( BK BX MN QN SI )

for boro in "${boroughs[@]}"
do
  rm ${boro}.csv
done
