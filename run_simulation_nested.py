import os
import pandas as pd
from pathlib import Path
import time
import argparse

from dataloader import *
from discrete_event_simu_nested import *

def main(sample_size, sample_id, k: int):
    """
    sample_size and sample_id marks the solution file to be used for simulation.
    """
    # DATA_DIR = Path(f'../SingleMarketData') 
    print(f'Testing allotion y indexed by {sample_size}Samples_{sample_id}')

    DATA_DIR = Path(f'../MultipleLegsData')  
    # DATA_DIR = Path(r"E:\WIP\RM\SSBLP\ToyExample")
    data_loader = AirlineDataLoader(k=k, data_dir=DATA_DIR)

    # Create output directory with sample info
    model = 'Nested_bigM'
    output_dir = DATA_DIR / f'{model}_performance_{k}T' / f'{sample_size}Samples'
    os.makedirs(output_dir, exist_ok=True)

    sol_df = pd.read_csv(
        DATA_DIR / f'nested_samples_output'
        / f'{sample_size}Samples/{sample_size}Samples_{sample_id}'
        / f'bigM_y_{sample_size}Samples_{sample_id}.csv', 
        low_memory=False)
    sol_df['Services'] = sol_df['Services'].astype(str)

    model_name=f"{model}_{sample_size}Samples_{sample_id}"

    simulated_revenue = run_all_samples_parallel_nested(
        data_loader=data_loader,
        protection_df=sol_df,
    )

    pd.Series({
        'revenue': simulated_revenue,
        }).to_csv(output_dir / f'{model_name}_simulated_revenue.csv')


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run SAA model with given sample size and ID.')
    parser.add_argument('--sample_size', type=int, default=5, help='Sample size for SAA')
    parser.add_argument('--sample_id', type=int, default=36, help='Sample ID to process')
    parser.add_argument('--k', type=int, default=1, help='Scaling factor for demand')
    
    args = parser.parse_args()
    
    # Run the main function with the provided arguments
    main(args.sample_size, args.sample_id, args.k)
