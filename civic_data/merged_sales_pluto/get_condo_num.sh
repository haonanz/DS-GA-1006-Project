#!/bin/bash

curl http://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/pad16c.zip > pad16c.zip

unzip -j pad16c.zip bobabbl.txt

rm pad16c.zip

python merge_sales_pluto.py

rm bobabbl.txt
