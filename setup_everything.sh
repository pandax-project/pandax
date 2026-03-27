#!/bin/bash

wget https://uofi.box.com/shared/static/9r1fgjdpoz113ed2al7k1biwxgnn9fpa -O dias_datasets.zip
# This should create a directory named dias-datasets
unzip dias_datasets.zip

cd dias-datasets
./copier.sh ../dias_notebooks
cd ../
rm -rf dias_datasets.zip dias-datasets

wget "https://drive.google.com/file/d/1cg89VLnr0bpB4w-xul-CirXj30MpdMEU/view?usp=drive_link" -O ds_datasets.zip
unzip ds_datasets.zip
cd ds_datasets
./copier.sh ../ds_notebooks
cd ../
rm -rf ds_datasets.zip ds_datasets

./verify_datasets_are_copied.sh