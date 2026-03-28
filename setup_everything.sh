#!/bin/bash

gdown 1h-jS2X7SmDw6i5zMYtqhR1-e0YGeyBF3
unzip datasets.zip
cd datasets
./copier.sh ..
cd ../
rm -rf datasets.zip datasets

./verify_datasets_are_copied.sh dias_notebooks ds_notebooks