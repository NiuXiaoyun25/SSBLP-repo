import numpy as np
import pandas as pd
from dataloader import AirlineDataLoader
from pathlib import Path


def simulate(
        data: AirlineDataLoader, 
        protection_df: pd.DataFrame,
        ):
    market = data.fares_df['Market'].unique()[0] 
    v0 = data.OA_attraction[market]

    sample_size = 3500

    # Note that this change the attribute ds_dict of data_loader
    ds_dict = data.compute_market_demand(
        filename=f'30T/{sample_size}Samples/ServicesDemand{sample_size}Samples_4.csv') 

    # Initialize sales
    revenue = 0

    np.random.seed(42)  # for reproducibility
    # both the protection level and total demand has rounding error
    for (mkt, sample_idx), demand in ds_dict.items():
        print(f'> Simulating sample {sample_idx}')
        print(f'  Market {mkt} with demand {demand}', end=', ')

        # Initialize inventory
        inventory = {
            (row.Market, row.Services, row.FareClass): row.Protection_Level
            for row in protection_df.itertuples()
            if row.Protection_Level > 0
        }

        sample_revenue = 0

        for customer_id in range(int(demand)):
            # Available products: positive inventory
            available_products = [p for p in inventory if inventory[p] > 0]

            if not available_products:
                # No products available, customer does not purchase
                break

            # Compute choice probabilities
            v_values = np.array([data.host_attraction[p] for p in available_products])
            denom = v0 + np.sum(v_values)
            product_probs = v_values / denom
            no_purchase_prob = v0 / denom

            probs = np.append(product_probs, no_purchase_prob)

            # Simulate customer choice
            # zero-based index for products, last index for no purchase
            choice = np.random.choice(len(probs), p=probs)

            if choice == len(probs) - 1:
                # Customer chooses not to purchase
                continue
            else:
                chosen_product = available_products[choice]
                revenue += data.r[chosen_product]
                sample_revenue += data.r[chosen_product]
                inventory[chosen_product] -= 1
        print(f'sample revenue: {sample_revenue}')
        
    simulated_revenue = revenue / sample_size

    print(f'Average revenue: {simulated_revenue }')

    return simulated_revenue

if __name__ == "__main__":
    DATA_DIR = Path(f'../SingleMarketData')
    data_loader = AirlineDataLoader(data_dir=DATA_DIR)
    protection_df = pd.read_csv(DATA_DIR / f"MILP_output/milp_solution.csv",
                                index_col=0,
                                dtype={'Services': str})
    simulate(
        data=data_loader,
        protection_df=protection_df
    )

