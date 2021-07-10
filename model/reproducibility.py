import random
import numpy as np
import torch

# This module is created to make our programs reproducible.
# To do this we control the seeds of all random sources.
# When will this break?
#     If the data provided to the program is not in the same order or the data file has been changed,
#     we cannot reproduce the results!!
# Note - We have seen small numerical differences e.g in the 15th decimal place even after controlling all
#        randomness. Please ignore these.

# This function needs to be the called before any function that uses randomness
def make_program_reproducible():
    seed = random.randint(0,2**32)
    print("Random seed =", seed)
    print("Replace `seed = random.randint(0,2**32)` by `seed = %d` to reproduce these results. (reproducibility.py - Line 15)" % (seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
