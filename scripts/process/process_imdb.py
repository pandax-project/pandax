# imdb-dataset-eda-project data needs special processing.
# you only need to run this if you just downloaded the data from dias, but can ignore if you are using an established repo.
import pandas as pd

# define what should be considered NA
na_values = ["\\N"]

# read in the two TSVs, coercing '\N' → NaN
df_ratings = pd.read_table(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-ratings.tsv",
    sep="\t",
    low_memory=False,
    na_values=na_values,
    keep_default_na=True,  # still catch pandas’ usual NA markers
)

df_meta = pd.read_table(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-metadata.tsv",
    sep="\t",
    low_memory=False,
    na_values=na_values,
    keep_default_na=True,
)

# tconst column looks like this: tt0000000
# we want to convert it to the integer value
df_meta["tconst"] = df_meta["tconst"].str.replace("tt", "").astype(int)

# now dump to CSV — by default pandas writes NaNs as empty fields,
# which read back in as NaN again
df_ratings.to_csv(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-ratings.csv",
    index=False,
)
df_meta.to_csv(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-metadata.csv",
    index=False,
)

df = pd.read_csv(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-ratings.csv"
)
df2 = pd.read_csv(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-metadata.csv"
)

df = df.drop(columns=["primaryTitle", "originalTitle"])
df2 = df2.drop(columns=["primaryTitle", "originalTitle"])

df.to_csv(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-ratings.csv",
    index=False,
)
df2.to_csv(
    "/home/dias-benchmarks/notebooks/sanket7994/imdb-dataset-eda-project/input/imdb-official-movies-dataset/title-metadata.csv",
    index=False,
)
