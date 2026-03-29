# PandaX

## Setup
To create the conda environment, run:
```bash
cd rapids-26.02
./recreate.sh "" /scratch/jieq/conda-envs/rapids-26.02 # with prefix.
./recreate.sh rapids-26.02 # without prefix.
```

Then, activate the environment:
```bash
conda activate /scratch/jieq/conda-envs/rapids-26.02
pip install gdown # TODO: move this inside recreate.sh
```

Then, copy the datasets:
```
./setup_everything.sh
```

Then verify the benchmarks for rewriting (small_bench.ipynb) and the actual benchmark (bench.ipynb) are the same besides for factors. 
```python
python scripts/verification/verify_bench.py --mode input 
```

Verify the CSV files through:
```python
python scripts/verification/verify_csv.py
```

Then process the CSV (optional, only run if your workflows fail)
```python
python scripts/process/process_csv.py dias_notebooks
python scripts/process/process_csv.py ds_notebooks
```

export root repo:
```bash
export PANDAX_ROOT=/scratch/jieq/pandax
```

clone elastic notebook repo:
```bash
git clone git@github.com:jieqiu0630/elastic-notebook.git
cd elastic-notebook
pip install -e .
```

Add both pandax repo and elastic-notebook repo to your `$PATH` env variable.

This repository contains the code for **PandaX**.

🚧 **Status:** Code will be released soon.  
📄 **Publication:** The paper that describes this project is currently under submission.

Meanwhile, see more information here:  
👉 https://pandax-project.github.io/
