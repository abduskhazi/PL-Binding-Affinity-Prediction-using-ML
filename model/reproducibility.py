import random
import numpy as np
import torch

# This module is created to make our programs reproducible.
# To do this we control the seeds of all random sources.
# When will this break?
#     If the data provided to the program is not in the same order or the data file has been changed,
#     we cannot reproduce the results!!

# This function needs to be the called before any function that uses randomness
def reproduce(seed = None):
    if seed == None:
        seed = random.randint(0,2**32)

    print("Execution ID =", seed)
    print("Use the above Execution ID to reproduce the results.")

    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 3)
