# DS-GA-1006-Project

### Capstone Project for NYU Data Science M.S.
### Authors: Benjamin Jakubowski and Haonan Zhou

### Project Objective: In this project, we aim to develop a model to:
1. Predict the sale price of residential properties in NYC
2. Identify real estate comps for a given property.


### Repo structure:
This repo contains scripts (primarily python and bash), plus exploratory ipython notebooks. The first set of directories contain scripts for getting and cleaning data. These include:
- civic_data: This directory contains scripts for getting and merging NYC civic datasets (PLUTO and annualized sales data). This approach ultimately was not taken in this project, but the scripts are retained for reference (and potential future use).
- streeteasy: This directory contains scripts for scraping the current sales posted on streeteasy. This was the initial approach taken to scraping streeteasy data, but it only produced 10k sale records, and none had labels for final sale price (only asking price was available).
- streeteasy_building: This directory contains an ipython notebook that explored scraping streeteasy sales by paging through buildings. This approach was not taken, but again the script was retained for reference.
- streeteasy_scrapy: This directory contains the python and bash scripts necessary to scrape the streeteasy sales pages (assumed to be integers from 0 to approximately 1400000 based on previous work). This was the approach ultimately taken to getting sales data.
- cartodb: This directory contains python scripts used for mapping using cartodb. The script used for the map in the final report was aggregate_by_nta.py, since we found pre-aggregating training set records was necessary (cartodb did not readily support online aggregation of 250k records).
- modeling: This directory contains python scripts for learning elastic-net regularized linear models (including models for sub-geographies), XGBoost models, and random forests. It also contains a subdirectory for LDA modeling of unit descriptions.

