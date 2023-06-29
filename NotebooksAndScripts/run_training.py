import os

# Parameters
variables = ["Hs"]#, "Tm02", "Dir"]
nruns = 1
nmodel = 1
# nmodel chooses one of the following architectures
models = {0: "Super-Resolution", 1: "Surrogate"}

for i in range(nruns):
    for var in variables:
        if nmodel == 0:
            cmd = f"python train_model.py {var}"
        elif nmodel == 1:
            cmd = f"python train_surrogate.py {var}"
        else:
            raise NotImplementedError("Invalid model choice.")
        os.system(cmd)
