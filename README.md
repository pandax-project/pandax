# PandaX

## Setup
To create the conda environment, run:
```bash
cd rapids-25.02
./recreate.sh "" /scratch/jieq/conda-envs/rapids-25.02 # with prefix.
./recreate.sh rapids-25.02 # without prefix.
```

Then, activate the environment:
```bash
conda activate rapids-25.02
pip install gdown # TODO: move this inside recreate.sh
```

Then, copy the datasets:
```
./setup_everything.sh
```

This repository contains the code for **PandaX**.

🚧 **Status:** Code will be released soon.  
📄 **Publication:** The paper that describes this project is currently under submission.

Meanwhile, see more information here:  
👉 https://pandax-project.github.io/
