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


""" MILPModel = MILPModel(
    model_name="MILP_model",
    output_dir=DATA_DIR / "MILP_output",
    data_loader=data_loader)
result = MILPModel.solve(sol_file="protection_level.csv", 
                              log_file="MILP_model.log",
                              solver_name="gurobi",
                              tee=True) """

""" nested_LP_model = NestedLP(
    model_name="nested_LP_model",
    output_dir=DATA_DIR / "nested_LP_output",
    data_loader=data_loader)

result = nested_LP_model.solve(sol_file="protection_level.csv", 
                              log_file="nested_LP_model.log",
                              solver_name="gurobi",
                              tee=True) """


""" bigM_model = NestedBigM(
    model_name="bigM_model",
    output_dir=DATA_DIR / "bigM_output",
    data_loader=data_loader)

result = bigM_model.solve(sol_file="protection_level.csv", 
                          log_file="bigM_model.log",
                          solver_name="gurobi",
                          tee=True) """

""" bigM_model = SampleBasedNestedBigM(
    model_name=f"bigM_{sample_size}Samples_{sample_id}",
    output_dir=DATA_DIR / f"bigM_samples_output/{sample_size}Samples/{sample_size}Samples_{sample_id}",
    data_loader=data_loader,
    sample_size=sample_size,
    sample_id=sample_id
)

result = bigM_model.solve(sol_file=f"bigM_y_{sample_size}Samples_{sample_id}.csv", 
                          log_file=f"bigM_{sample_size}Samples_{sample_id}.log",
                          solver_name="gurobi",
                          tee=True) """

""" y = pd.read_csv(r"D:\WIP\RM\MultipleLegsData\nested_LP_samples_output\nested_LP_samples_y.csv")
bigM_model.solve_fixed_y(y_values=y, tee=True, write_lp=True) """


""" saa = SAAModel(model_name="SAA_model",
               output_dir=DATA_DIR / "SAA_output",
               data_loader=data_loader)
result = saa.solve(sol_file="SAA_model.csv", 
                   log_file="SAA_model.log",
                   solver_name="gurobi",
                   tee=True) """


""" nested_LP_samples = SampleBasedNestedLP(
    model_name=f"nestedLP_{sample_size}Samples_{sample_id}",
    output_dir=DATA_DIR / f"nested_LP_samples_output/{sample_size}Samples/{sample_size}Samples_{sample_id}",
    data_loader=data_loader,
    sample_size=sample_size,
    sample_id=sample_id
) """

# y = pd.read_csv(r"D:\WIP\RM\MultipleLegsData\bigM_samples_output\protection_level.csv")
# nested_LP_samples.solve_fixed_y(y_values=y, tee=True, write_lp=True)

""" result = nested_LP_samples.solve(sol_file=f"nestedLP_{sample_size}Samples_{sample_id}.csv", 
                                 log_file=f"nestedLP_{sample_size}Samples_{sample_id}.log",
                                 solver_name="gurobi",
                                 tee=True) """
