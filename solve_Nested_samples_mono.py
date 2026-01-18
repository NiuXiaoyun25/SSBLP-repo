import pyomo.environ as pyo
from pyomo.environ import *

import os
import pandas as pd
from dataloader import *
from model import *
from nestedmodel import *

import argparse

parser = argparse.ArgumentParser(description="Solve nested model with sample-based data")
parser.add_argument("--sample_size", type=int, default=10, help="Number of samples")
parser.add_argument("--sample_id", type=int, default=1, help="Sample ID")
parser.add_argument("--k", type=int, default=1, help="Parameter k")

args = parser.parse_args()

sample_size = args.sample_size
sample_id = args.sample_id
k = args.k

# DATA_DIR = Path(f'../MultipleLegsData')
# DATA_DIR = Path(r"f:/WIP/RM/SSBLP/MultipleLegsData")
DATA_DIR = Path(f'../SSBLPdata')
data_loader = AirlineDataLoader(k=k, data_dir=DATA_DIR)

# Load the specific sample data
ds_dict = data_loader.compute_demand_samples(
    filename=f'{k}T/{sample_size}Samples/ServicesDemand{sample_size}Samples_{sample_id}.csv') 


bigM_model = SampleBasedNestedBigM_Mono(
    model_name=f"nestedbigMmono_{sample_size}Samples_{sample_id}",
    output_dir=DATA_DIR / f"nestedmono_samples_output/{sample_size}Samples/{sample_size}Samples_{sample_id}",
    data_loader=data_loader,
    sample_size=sample_size,
    sample_id=sample_id
)

result = bigM_model.solve(sol_file=f"nestedmono_y_{sample_size}Samples_{sample_id}.csv", 
                          log_file=f"nestedmono_{sample_size}Samples_{sample_id}.log",
                          solver_name="gurobi",
                          tee=True)
