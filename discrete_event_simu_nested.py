import multiprocessing as mp
import numpy as np
import pandas as pd
from dataloader import AirlineDataLoader
from pathlib import Path
from collections import defaultdict, OrderedDict
import copy
import os


# Define fare class order
FARE_CLASS_ORDER = list(chr(ord('A') + i) for i in range(8))

def _group_sum_cumulative(df: pd.DataFrame, value_col: str = 'Protection_Level'):
    """
    Generalized cumulative-sum grouping over (Market, Services) for FareClass thresholds.
    - value_col: column to sum (e.g., 'Protection_Level' or 'Sales').
    - services_col: target services column name; accepts input df with 'Service' and maps it to 'Services'.
    - fare_order: optional list specifying fare-class order (defaults to unique fare classes sorted reverse).
    Returns: dict mapping (market, services) -> OrderedDict(fare_class -> cumulative_sum).
    """
    if 'Service' in df.columns:
        df = df.rename(columns={'Service': 'Services'})

    grouped = {}

    for (mkt, svc), group in df.groupby(['Market', 'Services']):
        # initialize ordered dict for all fare classes in fare_order
        cum = OrderedDict((fc, 0) for fc in FARE_CLASS_ORDER)
        fare_levels = sorted(group['FareClass'].unique(), reverse=True)
        
        for fc in fare_levels:
            accumulative_value = group.loc[group['FareClass'].apply(lambda x: ord(x) >= ord(fc)), value_col].sum()
            cum[fc] = accumulative_value

        grouped[(mkt, svc)] = cum

    return grouped


# Backwards-compatible wrappers
def _group_sum_protection(protection_df: pd.DataFrame):
    return _group_sum_cumulative(protection_df, value_col='Protection_Level')


def _group_sum_sales(sales_df: pd.DataFrame):
    return _group_sum_cumulative(sales_df, value_col='Sales')


def simulate_one_sample_nested(data: AirlineDataLoader, protection_df: pd.DataFrame, sample_idx: int):
    print('\n')
    print(f'> Simulating nested sample {sample_idx}')
    # products = [(row.Market, row.Services, row.FareClass)
    #             for row in protection_df.itertuples()
    #             if row.Protection_Level > 0]  # THIS IS A BUG! The higher class can access lower class capacity!
    nested_caps = _group_sum_protection(protection_df)
    print(f'Initial nested capacities: {nested_caps}')
    products = [(mkt, svc, fc) for (mkt, svc), fare_dict in nested_caps.items() for fc in fare_dict.keys() if fare_dict[fc] > 0]

    # Deep copy the cumulative caps so we can decrement them as sales occur
    nested_caps_remaining = copy.deepcopy(nested_caps)
    
    simu_sample_size = 3500
    # Load raw demand dictionary and restructure: sample_idx → market → demand
    raw_ds_dict = data.compute_demand_samples(
        filename=f'{data.k}T/{simu_sample_size}Samples/ServicesDemand{simu_sample_size}Samples_1.csv')

    nested_ds_dict = defaultdict(dict)
    for (market, s), demand in raw_ds_dict.items():
        nested_ds_dict[s][market] = demand
    
    def nested_sale_feasible(market, service, fare):
        """
        Selling one unit of (market,service,fare) is feasible if every cumulative
        capacity that covers 'fare' has at least 1 remaining unit.
        (i.e., for every threshold fc where ord(fare) >= ord(fc), nested_caps_remaining[key][fc] >= 1)
        """
        key = (market, service)
        if key not in nested_caps_remaining:
            raise ValueError(f'Market {market} and service {service} not found!')

        # For the chosen fare, ensure every cumulative threshold that includes this fare has >= 1 remaining.
        for fc, remaining in nested_caps_remaining[key].items():
            if ord(fare) >= ord(fc) and remaining < 1:
                return False
        return True

    sales = 0
    sales_dict = defaultdict(dict)

    # iterate markets in this sample
    sample_revenue = 0
    print('-----' * 5, f'sample id {sample_idx}', '-----' * 5)
    for market, demand in nested_ds_dict[sample_idx].items():
        # print(f'Market: {market}, Demand: {demand}')
        v0 = data.OA_attraction[market]

        for customer_idx in range(int(demand)):
            print(f'> customer ID: {customer_idx + 1}')
            available_products = [
                p for p in products
                if p[0] == market and nested_sale_feasible(p[0], p[1], p[2])
            ]
            # print(f'Available products: {available_products}')
            if not available_products:
                break

            v_values = np.array([data.host_attraction[p] for p in available_products])
            denom = v0 + np.sum(v_values)
            product_probs = v_values / denom
            no_purchase_prob = v0 / denom

            probs = np.append(product_probs, no_purchase_prob)
            choice = np.random.choice(len(probs), p=probs)

            if choice == len(probs) - 1:
                continue
            else:
                sales += 1
                chosen_product = available_products[choice]
                # print(f'Chosen product: {chosen_product}')
                service = chosen_product[1]
                fare_class = str(chosen_product[2])
                
                # Initialize service dict with ordered fare classes if not exists
                if service not in sales_dict[market]:
                    sales_dict[market][service] = OrderedDict((fc, 0) for fc in FARE_CLASS_ORDER)
                sales_dict[market][service][fare_class] += 1
                
                sample_revenue += data.r[chosen_product]
                
                key = (chosen_product[0], chosen_product[1])

                for fc_key in list(nested_caps_remaining[key].keys()):
                    if ord(fare_class) >= ord(fc_key):
                        nested_caps_remaining[key][fc_key] -= 1
            # print(f'Nested capacities remaining: {nested_caps_remaining}')
                        

    # Convert sales_dict to DataFrame
    # rows = []
    # for market, services in sales_dict.items():
    #     for service, fare_classes in services.items():
    #         for fare_class, sale_count in fare_classes.items():
    #             rows.append({
    #                 'Market': market,
    #                 'Service': service,
    #                 'FareClass': fare_class,
    #                 'Sales': sale_count
    #             })
    # sales_df = pd.DataFrame(rows)

    # output_dir = Path(data.data_dir / f'Nested_bigM_performance_1T/5Samples_36')
    # os.makedirs(output_dir, exist_ok=True)
                      
    # sales_df.to_csv(output_dir / f'nested_sales_sample_{sample_idx}.csv', index=False)

    return sample_revenue


def run_all_samples_parallel_nested(data_loader: AirlineDataLoader, 
                                    protection_df: pd.DataFrame, 
                                    n_processes=32):
    args = [(data_loader, protection_df, idx) for idx in range(1, 3500 + 1)]

    with mp.Pool(processes=n_processes) as pool:
        sample_revenues = pool.starmap(simulate_one_sample_nested, args)

    total_revenue = sum(sample_revenues)
    num_simulated_samples = 3500
    avg_revenue = total_revenue / num_simulated_samples

    return avg_revenue


if __name__ == '__main__':
   # DATA_DIR = Path(f'../MultipleLegsData')  
    k = 1
    sample_size = 5
    sample_id = 36
    DATA_DIR = Path(r"E:\WIP\RM\SSBLP\ToyExample")
    data_loader = AirlineDataLoader(k=k, data_dir=DATA_DIR)

    # Create output directory with sample info
    model = 'Nested_bigM'
    output_dir = DATA_DIR / f'{model}_performance_{k}T' / f'{sample_size}Samples'
    os.makedirs(output_dir, exist_ok=True)

    sol_df = pd.read_csv(
        DATA_DIR / f'nested_samples_output'
        / f'{sample_size}Samples/{sample_size}Samples_{sample_id}'
        / f'nested_y_{sample_size}Samples_{sample_id}.csv', 
        low_memory=False)
    sol_df['Services'] = sol_df['Services'].astype(str)

    model_name=f"{model}_{sample_size}Samples_{sample_id}"

    for simu_sample_id in range(1, 2):
        simulated_revenue = simulate_one_sample_nested(
            data_loader,
            sol_df,
            simu_sample_id
        )

