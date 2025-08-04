import multiprocessing as mp
import numpy as np
import pandas as pd
from dataloader import AirlineDataLoader
from pathlib import Path
from collections import defaultdict


def simulate_one_sample(data: AirlineDataLoader, protection_df: pd.DataFrame, sample_idx: int):
    print(f'> Simulating sample {sample_idx}')

    # Initialize network-wide inventory
    network_inventory = {
        (row.Market, row.Services, row.FareClass): row.Protection_Level
        for row in protection_df.itertuples()
        if row.Protection_Level > 0
    } 
    
    inventory = network_inventory.copy()
    sample_revenue = 0

    sample_size = 3500

    # Load raw demand dictionary and restructure: sample_idx → market → demand
    raw_ds_dict = data.compute_market_demand(
        filename=f'{data.k}T/{sample_size}Samples/ServicesDemand{sample_size}Samples_4.csv')

    nested_ds_dict = defaultdict(dict)
    for (market, s), demand in raw_ds_dict.items():
        nested_ds_dict[s][market] = demand


    for market, demand in nested_ds_dict[sample_idx].items():
        print(f'+ Processing market: {market}, demand: {demand}')
        v0 = data.OA_attraction[market]

        for customer_idx in range(int(demand)):
            available_products = [p for p in inventory if p[0] == market and inventory[p] > 0]
    
            if not available_products:
                break

            v_values = np.array([data.host_attraction[p] for p in available_products])
            denom = v0 + np.sum(v_values)
            product_probs = v_values / denom
            no_purchase_prob = v0 / denom

            probs = np.append(product_probs, no_purchase_prob)
            choice = np.random.choice(len(probs), p=probs)

            if choice == len(probs) - 1:
                # Customer chooses not to purchase
                continue
            else:
                chosen_product = available_products[choice]
                sample_revenue += data.r[chosen_product]
                inventory[chosen_product] -= 1

    print(f'Sample {sample_idx} revenue: {sample_revenue}')
    return sample_revenue


def run_all_samples_parallel(data_loader: AirlineDataLoader, 
                             protection_df: pd.DataFrame, 
                             n_processes=32):
    args = [(data_loader, protection_df, idx) for idx in range(1, 3500 + 1)]

    with mp.Pool(processes=n_processes) as pool:
        sample_revenues = pool.starmap(simulate_one_sample, args)

    total_revenue = sum(sample_revenues)
    num_simulated_samples = 3500
    avg_revenue = total_revenue / num_simulated_samples

    print(f'Total revenue: {total_revenue}')
    print(f'Average revenue: {avg_revenue}')

    return avg_revenue