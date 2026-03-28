#!/bin/bash

gdown 1QhD03KWWo4dlx8EtHAi5t33G4_vXXr9_
unzip datasets.zip
cd datasets
./copier.sh ..
cd ../
rm -rf datasets.zip datasets

./verify_datasets_are_copied.sh dias_notebooks ds_notebooks