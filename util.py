import pandas as pd
import numpy as np

def rename_cols(val: str):
    # print(val)
    val = val.strip()
    # handle fare columns
    # if val.lower().startswith("fare "):
    # return "f" + val[5:]

    val = val.replace(" ", "_")
    return val