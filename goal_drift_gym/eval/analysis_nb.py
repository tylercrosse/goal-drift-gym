
# %%

import gymnasium as gym
import numpy as np
import pandas as pd
from IPython import get_ipython

# auto reload
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
