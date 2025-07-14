import pyomo.environ as pyo
from pyomo.environ import *
import os
import pandas as pd
from pathlib import Path
from pyomo.opt import TerminationCondition
import time
import argparse

from dataloader import *
from model import SAAModel
from discrete_event_simu import simulate

def main(sample_size, sample_id, k: int):
    DATA_DIR = Path(f'../SingleMarketData')
    data_loader = AirlineDataLoader(k=k, data_dir=DATA_DIR)

    # Load the specific sample data
    ds_dict = data_loader.compute_market_demand(
        filename=f'{k}T/{sample_size}Samples/ServicesDemand{sample_size}Samples_{sample_id}.csv') 
    
    # Create output directory with sample info
    saa = 'saa'
    output_dir = DATA_DIR / f'{saa}_output_{k}T' / f'{sample_size}Samples'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and solve the model
    saa_model = SAAModel(model_name=f"{saa}_{sample_size}Samples_{sample_id}",
                         output_dir=output_dir,
                         data_loader=data_loader)
    
    # Solve with unique output filenames
    sol_df = saa_model.solve(sol_file=f'sol_{sample_size}Samples_{sample_id}.csv', 
                             log_file=f'{saa}_{sample_size}Samples_{sample_id}.log')
        
    simulated_revenue = simulate(
        data=data_loader,
        protection_df=sol_df,
    )

    pd.Series({
        'revenue': simulated_revenue,
        }).to_csv(saa_model.output_dir / f'{saa_model.model.name}_simulated_revenue.csv')


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run SAA model with given sample size and ID.')
    parser.add_argument('--sample_size', type=int, required=True, help='Sample size for SAA')
    parser.add_argument('--sample_id', type=int, required=True, help='Sample ID to process')
    parser.add_argument('--k', type=int, required=True, help='Scaling factor for demand')
    
    args = parser.parse_args()
    
    # Run the main function with the provided arguments
    main(args.sample_size, args.sample_id, args.k)