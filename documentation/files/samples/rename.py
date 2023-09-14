import os
import copy

idx = 0
it = 0

for f in os.listdir():
    file = copy.copy(f)
    if file.split(".")[1] == "py" or file.split("-")[1].split(".")[0] in ["qv", "real", "predicted", "quantized"]:
        continue
    if file.find("real") > 0:
        idx = file.split("-")[-6]
        file = f"{idx}-real." + file.split(".")[1]
    elif file.find("qv") > 0:
        idx = file.split("-")[-6]
        file = f"{idx}-qv." + file.split(".")[1]
    elif file.find("quantized") > 0:
        idx = file.split("-")[-6]
        file = f"{idx}-quantized." + file.split(".")[1]
    else:
        idx = file.split("-")[-1].split(".")[0]
        file = f"{idx}-predicted." + file.split(".")[1]
    os.rename(f, file)
