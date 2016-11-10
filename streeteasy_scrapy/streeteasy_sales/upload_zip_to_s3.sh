#!/bin/bash

zip -r data.zip data

aws s3 cp data.zip s3://capstone-data-11-10-16/
