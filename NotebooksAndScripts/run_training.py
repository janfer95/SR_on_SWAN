import os

# Parameters
variables = ["Hs", "Tm02", "Dir"]
nruns = 1
# Choose 0 for super-resolution and 1 for surrogate model
nmodel = 0

for i in range(nruns):
    for var in variables:
        if nmodel == 0:
            cmd = f"python train_superresolution.py {var}"
        elif nmodel == 1:
            cmd = f"python train_surrogate.py {var}"
        else:
            raise NotImplementedError("Invalid model choice.")
        os.system(cmd)
