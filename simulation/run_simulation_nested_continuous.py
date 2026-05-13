import argparse
import copy
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataloader import AirlineDataLoader
from discrete_event_simu_nested import _group_sum_protection


EPS = 1e-9


def _demand_sample_file(k: int, sample_size: int, sample_id: int) -> Path:
    return Path(
        f"{k}T/{sample_size}Samples/"
        f"ServicesDemand{sample_size}Samples_{sample_id}.csv"
    )


def _load_market_demands(
    data: AirlineDataLoader,
    sample_size: int,
    sample_id: int,
):
    demand_file = _demand_sample_file(
        k=data.k,
        sample_size=sample_size,
        sample_id=sample_id,
    )
    raw_ds_dict = data.compute_demand_samples(
        filename=demand_file
    )

    demands_by_sample = defaultdict(dict)
    for (market, sample), demand in raw_ds_dict.items():
        demands_by_sample[sample][market] = float(demand)

    # The loader stores a defaultdict with a local default factory, which is
    # awkward to pickle on Windows multiprocessing after sample loading.
    data.ds_dict = dict(raw_ds_dict)

    return demands_by_sample, demand_file


def _nested_sale_feasible(nested_caps_remaining, market, service, fare):
    key = (market, service)
    if key not in nested_caps_remaining:
        raise ValueError(f"Market {market} and service {service} not found!")

    for fc, remaining in nested_caps_remaining[key].items():
        if ord(fare) >= ord(fc) and remaining <= EPS:
            return False
    return True


def _available_products(products, nested_caps_remaining, market):
    return [
        p
        for p in products
        if p[0] == market
        and _nested_sale_feasible(
            nested_caps_remaining=nested_caps_remaining,
            market=p[0],
            service=p[1],
            fare=p[2],
        )
    ]


def _product_probabilities(data: AirlineDataLoader, market, available_products):
    v0 = float(data.OA_attraction[market])
    v_values = np.array(
        [float(data.host_attraction[p]) for p in available_products],
        dtype=float,
    )
    denom = v0 + np.sum(v_values)
    if denom <= EPS:
        return {}

    return {
        product: float(v_value / denom)
        for product, v_value in zip(available_products, v_values)
        if v_value > EPS
    }


def _max_customer_mass_before_capacity_bind(
    product_probs,
    nested_caps_remaining,
):
    max_mass = float("inf")

    for (market, service), fare_dict in nested_caps_remaining.items():
        for fc, remaining in fare_dict.items():
            consumption_rate = sum(
                prob
                for product, prob in product_probs.items()
                if product[0] == market
                and product[1] == service
                and ord(product[2]) >= ord(fc)
            )
            if consumption_rate > EPS:
                max_mass = min(max_mass, remaining / consumption_rate)

    return max_mass


def _apply_fractional_sales(
    data: AirlineDataLoader,
    product_probs,
    customer_mass,
    nested_caps_remaining,
):
    revenue = 0.0

    for product, probability in product_probs.items():
        sale_amount = probability * customer_mass
        if sale_amount <= EPS:
            continue

        revenue += sale_amount * float(data.r[product])
        key = (product[0], product[1])
        fare_class = str(product[2])

        for fc_key in list(nested_caps_remaining[key].keys()):
            if ord(fare_class) >= ord(fc_key):
                remaining = nested_caps_remaining[key][fc_key] - sale_amount
                nested_caps_remaining[key][fc_key] = max(0.0, remaining)

    return revenue


def simulate_one_sample_nested_continuous(
    data: AirlineDataLoader,
    protection_df: pd.DataFrame,
    demands_by_sample,
    sample_idx: int,
):
    print("\n")
    print(f"> Simulating continuous nested sample {sample_idx}")

    nested_caps = _group_sum_protection(protection_df)
    print(f"Initial nested capacities: {nested_caps}")
    products = [
        (market, service, fare)
        for (market, service), fare_dict in nested_caps.items()
        for fare, capacity in fare_dict.items()
        if capacity > EPS
    ]
    nested_caps_remaining = copy.deepcopy(nested_caps)

    if sample_idx not in demands_by_sample:
        raise ValueError(f"Demand sample {sample_idx} not found in simulation samples.")

    sample_revenue = 0.0
    for market, demand in demands_by_sample[sample_idx].items():
        remaining_customers = float(demand)

        while remaining_customers > EPS:
            available_products = _available_products(
                products=products,
                nested_caps_remaining=nested_caps_remaining,
                market=market,
            )
            if not available_products:
                break

            product_probs = _product_probabilities(
                data=data,
                market=market,
                available_products=available_products,
            )
            if not product_probs:
                break

            max_customer_mass = _max_customer_mass_before_capacity_bind(
                product_probs=product_probs,
                nested_caps_remaining=nested_caps_remaining,
            )
            if max_customer_mass <= EPS:
                break

            customer_mass = min(remaining_customers, max_customer_mass)
            sample_revenue += _apply_fractional_sales(
                data=data,
                product_probs=product_probs,
                customer_mass=customer_mass,
                nested_caps_remaining=nested_caps_remaining,
            )
            remaining_customers -= customer_mass

    return sample_revenue


def run_all_samples_parallel_nested_continuous(
    data_loader: AirlineDataLoader,
    protection_df: pd.DataFrame,
    demand_sample_size: int,
    demand_sample_id: int,
    n_processes: int = 32,
):
    demands_by_sample, demand_file = _load_market_demands(
        data=data_loader,
        sample_size=demand_sample_size,
        sample_id=demand_sample_id,
    )
    sample_indices = list(range(1, demand_sample_size + 1))
    missing_samples = [idx for idx in sample_indices if idx not in demands_by_sample]
    if missing_samples:
        raise ValueError(
            f"Demand file {data_loader.data_dir / demand_file} does not contain "
            f"sample(s): {missing_samples}"
        )

    args = [
        (data_loader, protection_df, demands_by_sample, idx)
        for idx in sample_indices
    ]

    if n_processes == 1:
        sample_revenues = [
            simulate_one_sample_nested_continuous(*arg)
            for arg in args
        ]
    else:
        with mp.Pool(processes=n_processes) as pool:
            sample_revenues = pool.starmap(simulate_one_sample_nested_continuous, args)

    total_revenue = sum(sample_revenues)
    avg_revenue = total_revenue / demand_sample_size
    sample_revenue_df = pd.DataFrame(
        {
            "Sample": sample_indices,
            "Revenue": sample_revenues,
        }
    )

    return avg_revenue, sample_revenue_df, demand_file


def _read_mip_objective(solution_dir: Path, sample_size: int, sample_id: int):
    obj_file = solution_dir / f"SNestedmono_{sample_size}Samples_{sample_id}_obj.csv"
    if not obj_file.exists():
        return None

    obj_df = pd.read_csv(obj_file, index_col=0)
    return float(obj_df.loc["objective"].iloc[0])


def main(
    sample_size: int,
    sample_id: int,
    k: int,
):
    """
    sample_size and sample_id mark the y solution to simulate.
    The simulation reuses that exact demand sample file.
    """
    print(f"Testing continuous nested allocation y indexed by {sample_size}Samples_{sample_id}")

    demand_sample_size = sample_size
    demand_sample_id = sample_id
    print(
        "Using demand samples indexed by "
        f"{demand_sample_size}Samples_{demand_sample_id}"
    )

    # Keep the same data directory convention as solve_Nested_samples.py.
    DATA_DIR = Path(f'../SSBLPdata')
    data_loader = AirlineDataLoader(k=k, data_dir=DATA_DIR)

    model = "SNestedContinuous"
    output_dir = DATA_DIR / f"{model}_performance_{k}T" / f"{sample_size}Samples"
    os.makedirs(output_dir, exist_ok=True)

    solution_dir = (
        DATA_DIR / f"SNestedmono_output"
        / f"{sample_size}Samples/{sample_size}Samples_{sample_id}"
    )
    solution_file = solution_dir / f"Snestedmono_y_{sample_size}Samples_{sample_id}.csv"
    sol_df = pd.read_csv(
        solution_file,
        low_memory=False,
    )
    sol_df["Services"] = sol_df["Services"].astype(str)

    if (demand_sample_size, demand_sample_id) == (sample_size, sample_id):
        model_name = f"{model}_{sample_size}Samples_{sample_id}"
    else:
        model_name = (
            f"{model}_{sample_size}Samples_{sample_id}"
            f"_on_{demand_sample_size}Samples_{demand_sample_id}"
        )

    simulated_revenue, sample_revenue_df, demand_file = (
        run_all_samples_parallel_nested_continuous(
            data_loader=data_loader,
            protection_df=sol_df,
            demand_sample_size=demand_sample_size,
            demand_sample_id=demand_sample_id,
        )
    )
    mip_objective = _read_mip_objective(solution_dir, sample_size, sample_id)

    revenue_summary = {
        "revenue": simulated_revenue,
        "solution_file": str(solution_file),
        "demand_sample_file": str(DATA_DIR / demand_file),
        "solution_sample_size": sample_size,
        "solution_sample_id": sample_id,
        "demand_sample_size": demand_sample_size,
        "demand_sample_id": demand_sample_id,
        "num_simulated_samples": demand_sample_size,
    }
    if mip_objective is not None:
        revenue_summary["mip_objective"] = mip_objective
        revenue_summary["simulation_minus_mip_objective"] = (
            simulated_revenue - mip_objective
        )

    pd.Series(revenue_summary).to_csv(
        output_dir / f"{model_name}_simulated_revenue.csv"
    )
    sample_revenue_df.to_csv(
        output_dir / f"{model_name}_sample_revenues.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run continuous nested simulation with fractional customer choices."
    )
    parser.add_argument("--sample_size", type=int, default=5, help="Number of samples")
    parser.add_argument("--sample_id", type=int, default=4, help="Sample ID")
    parser.add_argument("--k", type=int, default=1, help="Parameter k")

    args = parser.parse_args()
    main(
        sample_size=args.sample_size,
        sample_id=args.sample_id,
        k=args.k,
    )
