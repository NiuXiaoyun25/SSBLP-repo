import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import beta
import random

import csv
from pathlib import Path
import argparse

def main(sample_size, sample_id, k):
    # single leg
    # DATA_DIR = Path(f'../SingleMarketData')
    # network case
    DATA_DIR = Path(f'../SSBLPdata')

    service_fares_and_demands_path = DATA_DIR / "ServiceFaresAndDmds.csv"
    demand_input = pd.read_csv(service_fares_and_demands_path)

    Market_List_path = DATA_DIR / "MarketList.csv"
    mktlist_input = pd.read_csv(Market_List_path, usecols=['Market','host_shr'])

    # Merge the market share information onto the demand input file
    demand_input = demand_input.merge(mktlist_input, on='Market', how='outer')

    # initialize the main dataframe with all the samples
    keys = ['Services', 'Market', 'host_shr', 'Class', 'Demand', 'Sample']
    services_dmds_sample = pd.DataFrame(columns=keys)

    # loop across the eight fare classes
    for f in range(8):
        if f == 0:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd A ",
                    " Var A ",
                ]
            ].assign(Dmd_A=k * demand_input[" Dmd A "], Var_A=k * demand_input[" Var A "]).copy()
            services_dmds["shape"] = (
                services_dmds['Dmd_A'] ** 2 / services_dmds["Var_A"]
            )
            services_dmds["scale"] = services_dmds["Var_A"] / services_dmds["Dmd_A"]
            services_dmds["Class"] = "A"

        elif f == 1:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd B ",
                    " Var B ",
                ]
            ].assign(Dmd_B=k * demand_input[" Dmd B "], Var_B=k * demand_input[" Var B "]).copy()
            services_dmds["shape"] = (
                services_dmds["Dmd_B"] ** 2 / services_dmds["Var_B"]
            )
            services_dmds["scale"] = services_dmds["Var_B"] / services_dmds["Dmd_B"]
            services_dmds["Class"] = "B"

        elif f == 2:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd C ",
                    " Var C ",
                ]
            ].assign(Dmd_C=k * demand_input[" Dmd C "], Var_C=k * demand_input[" Var C "]).copy()
            services_dmds["shape"] = (
                services_dmds["Dmd_C"] ** 2 / services_dmds["Var_C"]
            )
            services_dmds["scale"] = services_dmds["Var_C"] / services_dmds["Dmd_C"]
            services_dmds["Class"] = "C"

        elif f == 3:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd D ",
                    " Var D ",
                ]
            ].assign(Dmd_D=k * demand_input[" Dmd D "], Var_D=k * demand_input[" Var D "]).copy()
            services_dmds["shape"] = (
                services_dmds["Dmd_D"] ** 2 / services_dmds["Var_D"]
            )
            services_dmds["scale"] = services_dmds["Var_D"] / services_dmds["Dmd_D"]
            services_dmds["Class"] = "D"

        elif f == 4:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd E ",
                    " Var E ",
                ]
            ].assign(Dmd_E=k * demand_input[" Dmd E "], Var_E=k * demand_input[" Var E "]).copy()
            services_dmds["shape"] = (
                services_dmds["Dmd_E"] ** 2 / services_dmds["Var_E"]
            )
            services_dmds["scale"] = services_dmds["Var_E"] / services_dmds["Dmd_E"]
            services_dmds["Class"] = "E"

        elif f == 5:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd F ",
                    " Var F ",
                ]
            ].assign(Dmd_F=k * demand_input[" Dmd F "], Var_F=k * demand_input[" Var F "]).copy()
            services_dmds["shape"] = (
                services_dmds["Dmd_F"] ** 2 / services_dmds["Var_F"]
            )
            services_dmds["scale"] = services_dmds["Var_F"] / services_dmds["Dmd_F"]
            services_dmds["Class"] = "F"

        elif f == 6:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd G ",
                    " Var G ",
                ]
            ].assign(Dmd_G=k * demand_input[" Dmd G "], Var_G=k * demand_input[" Var G "]).copy()
            services_dmds["shape"] = (
                services_dmds["Dmd_G"] ** 2 / services_dmds["Var_G"]
            )
            services_dmds["scale"] = services_dmds["Var_G"] / services_dmds["Dmd_G"]
            services_dmds["Class"] = "G"

        else:
            services_dmds = demand_input[
                [
                    "Services",
                    "Market",
                    "LegOrig",
                    "LegDest",
                    "host_shr",
                    " Dmd H ",
                    " Var H ",
                ]
            ].assign(Dmd_H=k * demand_input[" Dmd H "], Var_H=k * demand_input[" Var H "]).copy()
            services_dmds["shape"] = (
                services_dmds["Dmd_H"] ** 2 / services_dmds["Var_H"]
            )
            services_dmds["scale"] = services_dmds["Var_H"] / services_dmds["Dmd_H"]
            services_dmds["Class"] = "H"

        # updated RMR 2/9/23
        # loop to generate multiple samples
        for i in range(sample_size):
            # draw a random demand from the Gamma distribution for the market, service, class considered
            # services_dmds['Demand'] = np.random.default_rng().gamma(services_dmds['shape'],services_dmds['scale'],services_dmds.shape[0])
            services_dmds["Demand"] = np.random.gamma(
                services_dmds["shape"], services_dmds["scale"], services_dmds.shape[0]
            )
            services_dmds["Sample"] = i + 1

            # append the new dataframe onto the full sample
            services_dmds_sample = pd.concat(
                [services_dmds_sample, services_dmds], ignore_index=True, sort=False, join='inner',
            )

    Market_demand_host = services_dmds_sample.groupby(by=['Sample', 'Market'], as_index=False).agg({'Demand':'sum'})

    services_file = DATA_DIR / "oamkt svc pair.csv"
    mkt_svc_pair = pd.read_csv(services_file, usecols=['Market', 'Services'])
    mktlist_input = mktlist_input.merge(mkt_svc_pair, on='Market')

    mktDmd_host_shr = Market_demand_host.merge(mktlist_input, on='Market', how='outer')

    mktDmd_host_shr['Demand'] = (1 - mktDmd_host_shr['host_shr']) / mktDmd_host_shr['host_shr'] * mktDmd_host_shr['Demand']

    mkt_oa = mktDmd_host_shr.copy()

    services_dmds_sample = pd.concat([services_dmds_sample, mkt_oa], ignore_index=True)
    services_dmds_sample.drop(columns=['host_shr'], inplace=True)
    
    output_path = DATA_DIR / f'{k}T/{sample_size}Samples'
    output_path.mkdir(parents=True, exist_ok=True)
    services_dmds_sample.to_csv(
        output_path / f"ServicesDemand{sample_size}Samples_{sample_id}.csv", index=False)

    print(f"CSV file 'ServicesDemand{sample_size}Samples_{sample_id}.csv' has been successfully created in {DATA_DIR}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Generate demand samples with given sample size and ID.')
    parser.add_argument('--sample_size', type=int, required=True, help='Sample size')
    parser.add_argument('--sample_id', type=int, required=True, help='Sample ID')
    parser.add_argument('--k', type=int, required=True, help='Scaling factor for demand')
    
    args = parser.parse_args()
    
    # Run the main function with the provided arguments
    main(args.sample_size, args.sample_id, args.k)
