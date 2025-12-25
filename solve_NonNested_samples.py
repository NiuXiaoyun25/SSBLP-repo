import pyomo.environ as pyo
from pyomo.environ import *

import os
import pandas as pd
from dataloader import *
from model import *
from nestedmodel import *

import argparse

# Example usage
parser = argparse.ArgumentParser(description="Solve nested model with sample-based data")
parser.add_argument("--sample_size", type=int, default=2, help="Number of samples")
parser.add_argument("--sample_id", type=int, default=3, help="Sample ID")
parser.add_argument("--k", type=int, default=1, help="Parameter k")

args = parser.parse_args()

sample_size = args.sample_size
sample_id = args.sample_id
k = args.k

# DATA_DIR = Path(f'../MultipleLegsData')
DATA_DIR = Path(r"E:\WIP\RM\SSBLP\MultipleLegsData")
# DATA_DIR = Path(f'../SSBLPdata')
# DATA_DIR = Path(r"E:\WIP\RM\SSBLP\ToyExample")
data_loader = AirlineDataLoader(k=k, data_dir=DATA_DIR)

# Load the specific sample data
ds_dict = data_loader.compute_demand_samples(
    filename=f'{k}T/{sample_size}Samples/ServicesDemand{sample_size}Samples_{sample_id}.csv') 

nonnested_model = SAAModel(model_name="nonnested_model",
                           output_dir=DATA_DIR / f"NonNested_bigM_output/{sample_size}Samples/{sample_size}Samples_{sample_id}",
                           data_loader=data_loader)

nonnested_model.solve(sol_file=f"nonnested_{sample_size}Samples_{sample_id}.csv", 
                      log_file=f"nonnested_{sample_size}Samples_{sample_id}.log",
                      solver_name="gurobi",
                      tee=True)
