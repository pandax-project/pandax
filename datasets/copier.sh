#!/bin/bash

NB_ROOT=$1

ds_nbs=(
  "imdb"
  "nyc-flight"
  "nyc-taxi"
  "nyc-airbnb"
  "us-birth"
)

dias_nbs=(
  "lextoumbourou/feedback3-eda-hf-custom-trainer-sift"
  "paultimothymooney/kaggle-survey-2022-all-results"
  "dataranch/supermarket-sales-prediction-xgboost-fastai"
  "kkhandekar/environmental-vs-ai-startups-india-eda"
  "ampiiere/animal-crossing-villager-popularity-analysis"
  "aieducation/what-course-are-you-going-to-take"
  "saisandeepjallepalli/adidas-retail-eda-data-visualization"
  "joshuaswords/netflix-data-visualization"
  "spscientist/student-performance-in-exams"
  "ibtesama/getting-started-with-a-movie-recommendation-system"
  "nickwan/creating-player-stats-using-tracking-data"
  "erikbruin/nlp-on-student-writing-eda"
  "madhurpant/beautiful-kaggle-2022-analysis"
  "pmarcelino/comprehensive-data-exploration-with-python"
  "gksriharsha/eda-speedtests"
  "mpwolke/just-you-wait-rishi-sunak"
  "sanket7994/imdb-dataset-eda-project"
  "roopacalistus/retail-supermarket-store-analysis"
  "sandhyakrishnan02/indian-startup-growth-analysis"
  "josecode1/billionaires-statistics-2023"
)

for nb in "${ds_nbs[@]}"; do
  echo "ds_notebooks/${nb}"
  mkdir -p "${NB_ROOT}/ds_notebooks/${nb}"
  cp -r "${nb}/input" "${NB_ROOT}/ds_notebooks/${nb}/"
done

for nb in "${dias_nbs[@]}"; do
  echo "dias_notebooks/${nb}"
  mkdir -p "${NB_ROOT}/dias_notebooks/${nb}"
  cp -r "${nb}/input" "${NB_ROOT}/dias_notebooks/${nb}/"
done

