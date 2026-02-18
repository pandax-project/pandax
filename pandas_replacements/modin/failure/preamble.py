# This file is loaded as is in other files.

import ray

# Configure Modin and Ray
from modin.config import CpuCount, Engine

# Settings
NUM_CPUS = 12
ENGINE = "ray"

Engine.put(ENGINE)
CpuCount.put(NUM_CPUS)

assert ENGINE == "ray"

# NOTE: We suppress ray warnings
ray.init(
    num_cpus=NUM_CPUS,
    runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}},
    log_to_driver=False,
)

# Import modules
