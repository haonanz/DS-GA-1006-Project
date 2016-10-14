#!/bin/bash

years=( 2011 2012 2013 2014 2015 )
boroughs=( bronx brooklyn queens statenisland manhattan )

for boro in "${boroughs[@]}"
do
  for year in "${years[@]}"
  do
    curl https://www1.nyc.gov/assets/finance/downloads/pdf/rolling_sales/annualized-sales/${year}/${year}_${boro}.xls > ${year}_${boro}.xls
  done
done

file="all_residential_2011_2015.csv"

if [ -f $file ] ; then
  rm $file
fi


python make_residential_csv.py

for boro in "${boroughs[@]}"
do
  for year in "${years[@]}"
  do
    rm ${year}_${boro}.xls
  done
done
