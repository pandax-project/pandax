#!/bin/bash

wget https://uofi.box.com/shared/static/9r1fgjdpoz113ed2al7k1biwxgnn9fpa -O dias_datasets.zip
# This should create a directory named dias-datasets
unzip dias_datasets.zip

cd dias-datasets
./copier.sh ../dias_notebooks
cd ../
rm -rf dias_datasets.zip dias-datasets

gdown 1h-jS2X7SmDw6i5zMYtqhR1-e0YGeyBF3
unzip ds_datasets.zip
cd ds_datasets
./copier.sh ../ds_notebooks
cd ../
rm -rf ds_datasets.zip ds_datasets

./verify_datasets_are_copied.sh