#!/bin/bash

echo 'Running python code to learn LDA with sklearn...\n'
python learn_K_LDA.py
echo 'Running R code for SLDA...\n'
RScript sLDA.R
echo 'Making learning curve for LDA/sLDA K grid search...\n'
python make_plots_from_models.py
echo 'Based on learning curve, learning K = 10 sLDA representation...\n'
python final_sLDA_features.py
RScript sLDA_10.R
python GBR_for_sLDA_10.py
gzip -c final_test_ids.txt > ./../data/test_saleids.txt.gz
