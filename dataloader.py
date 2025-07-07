from pathlib import Path
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import pandas as pd
import numpy as np

from util import rename_cols

class AirlineDataLoader:
    """
    AirlineDataLoader is a class for loading, processing, and managing airline-related data from various CSV files.
    It provides methods to read fares, capacities, host and OA attraction values, mean market demand, and demand samples,
    and organizes them into convenient data structures for further analysis and modeling.
    Attributes:
        data_dir (Path): Directory containing the data files.
        k (int): Scaling factor for capacities and mean market demand.
        fares_df (pd.DataFrame): Processed fares data in long format.
        products (list): List of tuples representing (Market, Services, fare_class) products.
        r (dict): Dictionary mapping (Market, Services, fare_class) to fare values.
        flights (pd.DataFrame): Processed flight capacity data.
        host_attraction_df (pd.DataFrame): Raw host attraction data.
        host_attraction (dict): Dictionary mapping (Market, Services, Class) to mean host demand.
        OA_attraction_df (pd.DataFrame): Raw OA attraction data.
        OA_attraction (dict): Dictionary mapping Market to mean OA demand.
        mean_market_demand_df (pd.DataFrame): Raw mean market demand data.
        mean_market_demand (dict): Dictionary mapping Market to scaled mean market demand.
        demand_samples (pd.DataFrame): Loaded demand samples.
        num_scenarios (int): Number of unique demand scenarios.
        ds_dict (defaultdict): Dictionary mapping (Market, Sample) to computed demand.
    Methods:
        __init__(data_dir, fares_file, capacity_file, host_attraction_file, OA_attraction_file, mean_mkt_demand_file, k):
            Initializes the loader and loads all relevant data files.
        _load_and_process_fares(filename):
            Loads and processes service fares data, reshaping and extracting product information.
        _load_and_process_capacities(filename):
            Loads and processes flight capacity data, scaling by factor k.
        _load_host_attraction_values(filename):
            Loads and processes host attraction values from a CSV file.
        _load_OA_attraction_values(filename):
            Loads and processes OA attraction values from a CSV file.
        _load_mean_market_demand(filename):
            Loads and processes mean market demand data, scaling by factor k.
        read_demand_samples(filename):
            Loads demand samples from a CSV file and stores the number of scenarios.
        compute_market_demand(filename):
            Computes total market demand from demand samples, combining host and OA demand,
            and stores the result in a dictionary for easy querying.
    """
    def __init__(
        self,
        data_dir: Path,
        fares_file: str = "ServiceFaresAndDmds.csv",
        capacity_file: str = "FlightRmngCapacity.csv",
        host_attraction_file: str = "mean_host_demand.csv",
        OA_attraction_file: str = "mean_oa_demand.csv",
        mean_mkt_demand_file: str = "mean_market_demand.csv",
        k=30
    ):
        self.data_dir = data_dir
        self.k = k
        self._load_and_process_fares(fares_file)
        self._load_and_process_capacities(capacity_file)
        self._load_host_attraction_values(host_attraction_file)
        self._load_OA_attraction_values(OA_attraction_file)
        self._load_mean_market_demand(mean_mkt_demand_file)


    def _load_and_process_fares(self, filename: str):
        """Load and process service fares data"""
        # Load data
        df = pd.read_csv(
            self.data_dir / filename,
            dtype={
                'Market': 'str',
                'Services': 'str',
            }
        ).rename(mapper=rename_cols, axis="columns")


        # melt fare columns over
        fare_cols = [c for c in df.columns if "Fare" in c]
        cols_except_fares = [c for c in df.columns if c not in fare_cols]
        mdf = df.melt(
            id_vars=cols_except_fares,
            value_vars=fare_cols,
            var_name="fare_class",
            value_name="fare_val",
        )
        # rename fare_class
        temp_df = (
            mdf["fare_class"].str.replace("Fare_", "").str.split("_", expand=True)
        )
        mdf["fare_class"] = temp_df
        # single maarket
        mdf['f1'] = mdf["Services"].str.split("-", n=2, expand=True)
        # multi-market
        # mdf[['f1', 'f2']] = mdf["Services"].str.split("-", n=2, expand=True)
 
        self.fares_df = mdf
  
        # Create products list and value dictionaries
        self.products = [
            (row.Market, row.Services, row.fare_class) 
            for row in self.fares_df.itertuples()
        ]

        self.r = {(row.Market, row.Services, row.fare_class): row.fare_val
                 for row in self.fares_df.itertuples()}
        

    def _load_and_process_capacities(self, filename: str):
        """Load and process flight capacity data"""
        # Load data
        self.flights = pd.read_csv(
            self.data_dir / filename,
            dtype={'Flights': 'str', 'CapRmng': 'int32'}
        ).set_index('Flights').assign(CapRmng=lambda x: x['CapRmng'] * self.k)
        print(self.flights)

    def _load_host_attraction_values(self, filename: str):
        """Load and process host attraction values from a CSV file"""
        # Load data
        self.host_attraction_df = pd.read_csv(
            self.data_dir / filename,
            dtype={'Market': 'str', 'Services': 'str', 'Class': 'str', 'mean_host_demand': 'float32'}
        )
        # Process data
        self.host_attraction = {(row.Market, row.Services, row.Class): row.mean_host_demand 
                                for row in self.host_attraction_df.itertuples()}


    def _load_OA_attraction_values(self, filename: str):
        """Load and process OA attraction values from a CSV file"""
        # Load data
        self.OA_attraction_df = pd.read_csv(
            self.data_dir / filename,
            dtype={'Market': 'str', 'mean_oa_demand': 'float32'}
        )
        # Process data
        self.OA_attraction = {
            row.Market: row.mean_oa_demand 
            for row in self.OA_attraction_df.itertuples()
        }
        
        
    def _load_mean_market_demand(self, filename: str):
        """Load and process mean market demand data"""
        # Load data
        self.mean_market_demand_df = pd.read_csv(
            self.data_dir / filename,
            dtype={'Market': 'str', 'mean_market_demand': 'float32'}
        )
        
        # Process data
        self.mean_market_demand = {
            row.Market: row.mean_market_demand * self.k
            for row in self.mean_market_demand_df.itertuples()
        }

        print(f'after scaling mean market demand by k={self.k}:')
        print(self.mean_market_demand)
    

    def read_demand_samples(self, filename: Path) -> pd.DataFrame:

        df = pd.read_csv(self.data_dir / filename).rename(mapper=rename_cols, axis="columns")

        self.demand_samples = df
        self.num_scenarios = df['Sample'].nunique()


    def compute_market_demand(self, filename: Path):
        # Compute market demand from demand samples
        self.read_demand_samples(filename)
 
        samples_oa = self.demand_samples[pd.isna(self.demand_samples["Class"])].copy()
        samples_host = self.demand_samples[~pd.isna(self.demand_samples["Class"])].copy()
        samples_host_sum = samples_host.groupby(by=["Market", "Sample"]).sum('Demand').reset_index()

        merged = samples_oa.merge(samples_host_sum, on=["Market", "Sample"], how="left")

        merged["Demand"] = merged["Demand_x"] + merged["Demand_y"]
        DS = merged[["Sample", "Market", "Demand"]]
   
        # build dictionaries with these for easier querying
        def value_e():
            raise ValueError("200")

        ds_dict = defaultdict(
            value_e, {(t.Market, t.Sample): t.Demand for t in DS.itertuples()}
        )
        
        # Store the dictionary in the instance variable
        self.ds_dict = ds_dict

        return ds_dict


if __name__ == "__main__":
    # Set up the data loader
    DATA_DIR = Path(f'../SingleMarketData')    # single market case
    # DATA_DIR = Path(f'../SSBLPdata')   # network case
    data_loader = AirlineDataLoader(data_dir=DATA_DIR)
    sample_size = 10
    sample_id = 4

    # Load the specific sample data
    ds_dict = data_loader.compute_market_demand(
        filename=f'30T/{sample_size}Samples/ServicesDemand{sample_size}Samples_{sample_id}.csv') 

