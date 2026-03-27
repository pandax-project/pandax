import os

# STEFANOS: Conditionally import Modin Pandas
if "IREWR_WITH_MODIN" in os.environ and os.environ["IREWR_WITH_MODIN"] == "True":
    # STEFANOS: Import Modin Pandas
    import os

    os.environ["MODIN_ENGINE"] = "ray"
    import ray

    ray.init(
        num_cpus=int(os.environ["MODIN_CPUS"]),
        runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}},
    )
else:
    # STEFANOS: Import regular Pandas
    pass

import cudf as cd
import nvtx

passmark = 40
df = cd.read_csv(
    "/home/dias-benchmarks/notebooks/spscientist/student-performance-in-exams/input/StudentsPerformance.csv"
)
df = cd.concat([df] * 1000)
factor = 100
df = cd.concat([df] * factor)
df.info()
### cell 0 ###
_ = df.isnull().sum()

### cell 1 ###

df["math score"] = (
    df["math score"].astype(float).fillna(-1)
)  # Replace NaNs with a default value
# Vectorized conditional assignment using cuDF
df["Math_PassStatus"] = (
    (df["math score"] >= passmark).astype("str").replace({"True": "P", "False": "F"})
)
# Use cuDF's optimized value_counts()
_ = df["Math_PassStatus"].value_counts()

### cell 2 ###
df["reading score"] = (
    df["reading score"].astype(float).fillna(-1)
)  # Replace NaNs if needed
# Use cuDF's vectorized conditional assignment
df["Reading_PassStatus"] = (
    (df["reading score"] >= passmark).astype("str").replace({"True": "P", "False": "F"})
)
# Use cuDF's value_counts() (faster than pandas on GPU)
_ = df["Reading_PassStatus"].value_counts()

### cell 3 ###
df["writing score"] = df["writing score"].astype("float32")
# Use cuDF's boolean indexing + direct assignment (avoids unnecessary operations)
df["Writing_PassStatus"] = "F"  # Default all to 'F' first
df.loc[df["writing score"] >= passmark, "Writing_PassStatus"] = (
    "P"  # Assign 'P' where needed
)
# Efficient value_counts() with cuDF
_ = df["Writing_PassStatus"].value_counts()

### cell 4 ###
df["OverAll_PassStatus"] = (
    (
        (df["Math_PassStatus"] == "P")
        & (df["Reading_PassStatus"] == "P")
        & (df["Writing_PassStatus"] == "P")
    )
    .astype("str")
    .replace({"True": "P", "False": "F"})
)

# Optimized cuDF value_counts()
_ = df["OverAll_PassStatus"].value_counts()

with nvtx.annotate("cell_5"):
    ### cell 5 ###
    df["Total_Marks"] = df["math score"] + df["reading score"] + df["writing score"]
    df["Percentage"] = df["Total_Marks"] / 3
