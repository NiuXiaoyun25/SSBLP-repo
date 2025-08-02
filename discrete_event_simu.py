import numpy as np
import pandas as pd
from dataloader import AirlineDataLoader
from pathlib import Path
from collections import defaultdict

def simulate_network(data: AirlineDataLoader, protection_df: pd.DataFrame):
    sample_size = 3500

    # Load raw demand dictionary and restructure: sample_idx → market → demand
    raw_ds_dict = data.compute_market_demand(
        filename=f'{data.k}T/{sample_size}Samples/ServicesDemand{sample_size}Samples_4.csv')

    nested_ds_dict = defaultdict(dict)
    for (market, sample_idx), demand in raw_ds_dict.items():
        nested_ds_dict[sample_idx][market] = demand

    for sample_idx in nested_ds_dict:
        print(f'Sample {sample_idx} has {len(nested_ds_dict[sample_idx])} markets, total demand: {sum(nested_ds_dict[sample_idx].values())}')

    # Initialize network-wide inventory
    network_inventory = {
        (row.Market, row.Services, row.FareClass): row.Protection_Level
        for row in protection_df.itertuples()
        if row.Protection_Level > 0
    }

    np.random.seed(42)
    total_revenue = 0

    for sample_idx in range(1, sample_size+1):
        print(f'> Simulating sample {sample_idx}')
        inventory = network_inventory.copy()
        sample_revenue = 0

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

        print(f'Sample revenue: {sample_revenue}')
        total_revenue += sample_revenue
        break

    avg_revenue = total_revenue / sample_size
    print(f'Total revenue: {total_revenue}')
    print(f'Average revenue: {avg_revenue}')
    return avg_revenue

if __name__ == "__main__":
    DATA_DIR = Path(f'../SSBLPdata')
    k = 30
    data_loader = AirlineDataLoader(k=k, data_dir=DATA_DIR)
    protection_df = pd.read_csv(DATA_DIR / f"MILP_output/milp_solution.csv",
                                index_col=0,
                                dtype={'Services': str})
    simulate_network(
        data=data_loader,
        protection_df=protection_df
    )
