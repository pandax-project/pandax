#!/bin/bash

NB_ROOT=$1

nbs=(
    "imdb"
    "nyc-flight"
    "nyc-taxi"
    "nyc-airbnb"
    "us-birth"
)

for nb in ${nbs[@]}; do
  echo ${nb}
  cp -r ${nb}/input ${NB_ROOT}/${nb}/
done
