import os

# Parameters
variables = ["Hs", "Tm02", "Dir"]
nruns = 1
nmodel = 0
# nmodel chooses one of the following architectures
models = {0: "OriginalFukami", 1: "Subpixel",
          2: "SubpixelDilated"}

for i in range(nruns):
    for var in variables:
        cmd = f"python Train_Model.py {models[nmodel]} {var}"
        os.system(cmd)
