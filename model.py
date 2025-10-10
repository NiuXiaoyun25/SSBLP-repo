from xml.parsers.expat import model
import pyomo.environ as pyo
from pyomo.environ import *
import os
import pandas as pd
from pathlib import Path
from pyomo.opt import TerminationCondition

import time

from dataloader import *


class MILPModel: 
    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        self._calculate_model_parameters()
        self.model = self._build_model(model_name)
        self.output_dir = output_dir

        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)


    def _calculate_model_parameters(self):
        """Calculate derived model parameters"""
        self.M = max(self.data.mean_market_demand.values()) / min(self.data.host_attraction.values())

    def _cap_cons_lhs(self, flt: str, y):
        if 'f2' in self.data.fares_df.columns and not self.data.fares_df['f2'].isna().all():
            # Network case: flight can appear in f1 or f2
            df_flt = self.data.fares_df.query("f1 == @flt | f2 == @flt")
        else:
            # Single leg case: flight only appears in f1
            df_flt = self.data.fares_df.query('f1 == @flt')
        
        lhs = sum(y[t.Market, t.Services, t.fare_class] for t in df_flt.itertuples())
        return lhs

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.name = name
        
        # Sets
        model.I = pyo.Set(initialize=self.data.fares_df['Market'].unique().tolist())
        model.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        model.N = pyo.Set(initialize=self.data.products)

        # Variables
        model.y = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        model.z = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        model.z0 = pyo.Var(model.I, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.N, domain=pyo.Binary)
      
        # Objective
        def objective_rule(model):
            return sum(model.z[i, j, k] * self.data.r[i, j, k] 
                       for (i, j, k) in model.N)
        model.obj = pyo.Objective(rule=objective_rule, sense=maximize)
        
        # Constraints
        def demand_balance_rule(model, i):
            return (self.data.OA_attraction[i] * model.z0[i]
                    + sum(model.z[mkt, j, k] for (mkt, j, k) in model.N if mkt == i)
                    == self.data.mean_market_demand[i])
        model.demand_balance = pyo.Constraint(model.I, rule=demand_balance_rule)
        
        def sales_upper_rule1(model, i, j, k):
            return model.z[i, j, k] <= model.y[i, j, k]
        model.sales_upper1 = pyo.Constraint(model.N, rule=sales_upper_rule1)

        def sales_upper_rule2(model, i, j, k):
            return model.z[i, j, k] <= model.z0[i] * self.data.host_attraction[i, j, k]
        model.sales_upper2 = pyo.Constraint(model.N, rule=sales_upper_rule2)

        def bigM_rule1(model, i, j, k):
            return (model.z[i, j, k] >= model.y[i, j, k] - self.M * (1 - model.g[i, j, k]))
        model.bigM1 = pyo.Constraint(model.N, rule=bigM_rule1)

        def bigM_rule2(model, i, j, k):
            return model.z[i, j, k] >=model.z0[i] * self.data.host_attraction[i, j, k] - self.M * model.g[i, j, k]
        model.bigM2 = pyo.Constraint(model.N, rule=bigM_rule2)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        model.capacity_const = pyo.Constraint(model.L, rule=capacity_rule)

        return model


    """ def output_model(self, lp_file_loc='model_formulation.txt'):
        with open(lp_file_loc, 'w') as f:
            self.model.pprint(ostream=f) """


    def solve(self, sol_file: str, log_file: str, solver_name='gurobi', tee=True):
        solver = pyo.SolverFactory(solver_name)
        options = {
            'LogFile': str(self.output_dir / log_file),
            'LogToConsole': 1 if tee else 0
        }

        results = solver.solve(self.model, tee=tee, options=options)

        if results.solver.termination_condition == TerminationCondition.optimal:
            solution = {
                'objective': pyo.value(self.model.obj),
                'protection_levels': {(i, j, k): self.model.y[i, j, k].value
                                      for (i, j, k) in self.model.N}
            }


            sol_df = pd.DataFrame.from_dict(
                solution['protection_levels'],
                orient='index',
                columns=['Protection_Level']
            )

            sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])

            sol_df.reset_index(inplace=True)
            sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']

            sol_df.to_csv(self.output_dir / sol_file)

            z_sol = pd.DataFrame.from_dict(
                {(i, j, k): self.model.z[i, j, k].value for (i, j, k) in self.model.N},
                orient='index',
                columns=['Sales']
            )
            z_sol.index = pd.MultiIndex.from_tuples(z_sol.index, names=['Market', 'Services', 'FareClass'])
            z_sol.reset_index(inplace=True)
            z_sol.columns = ['Market', 'Services', 'FareClass', 'Sales']
            z_sol.to_csv(self.output_dir / f'{self.model.name}_zsol.csv')

            g_sol = pd.DataFrame.from_dict(
                {(i, j, k): self.model.g[i, j, k].value for (i, j, k) in self.model.N},
                orient='index', 
                columns=['Binary_G']
            )
            g_sol.index = pd.MultiIndex.from_tuples(g_sol.index, names=['Market', 'Services', 'FareClass'])
            g_sol.reset_index(inplace=True)
            g_sol.columns = ['Market', 'Services', 'FareClass', 'Binary_G']
            g_sol.to_csv(self.output_dir / f'{self.model.name}_gsol.csv')

            pd.Series(solution['objective'], name='obj').to_csv(self.output_dir / f'{self.model.name}_obj.csv')

            return {
                'solution': solution,
                'protection_levels_df': sol_df,
            }
        
        return None


class SAAModel:
    """SAA model for choice-based demand with stockout substitution (MILP)."""
    
    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        self._calculate_model_parameters()   # big-M
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        self.output_dir = output_dir
        self.model = self._build_model(model_name)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def _calculate_model_parameters(self):
        """Calculate derived model parameters"""
        self.M = max(self.data.ds_dict.values())

    def _cap_cons_lhs(self, flt: str, y):
        if 'f2' in self.data.fares_df.columns and not self.data.fares_df['f2'].isna().all():
            # Network case: flight can appear in f1 or f2
            df_flt = self.data.fares_df.query("f1 == @flt | f2 == @flt")
        else:
            # Single leg case: flight only appears in f1
            df_flt = self.data.fares_df.query('f1 == @flt')
        
        lhs = sum(y[t.Market, t.Services, t.fare_class] for t in df_flt.itertuples())
        return lhs

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.name = name

        # Sets
        model.I = pyo.Set(initialize=self.data.fares_df['Market'].unique().tolist())
        model.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        model.N = pyo.Set(initialize=self.data.products)
        model.S = pyo.Set(initialize=range(1, self.data.num_scenarios + 1))

        # Variables
        model.y = pyo.Var(model.N, domain=pyo.NonNegativeReals)  
        model.z = pyo.Var(model.N, model.S, domain=pyo.NonNegativeReals)          
        model.z0 = pyo.Var(model.I, model.S, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.N, model.S, domain=pyo.Binary)     

        # Objective: Maximize average revenue
        def objective_rule(model):
            return (1 / self.data.num_scenarios) * sum(
                model.z[i, j, k, s] * self.data.r[i, j, k] 
                for (i, j, k) in model.N
                for s in model.S
            )
        model.obj = pyo.Objective(rule=objective_rule, sense=maximize)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        model.capacity_const = pyo.Constraint(model.L, rule=capacity_rule)   

        # Demand balance
        def demand_balance_rule(model, i, s):
            return (
                    self.data.OA_attraction[i] * model.z0[i, s] 
                    + sum(model.z[mkt, j, k, s]
                          for (mkt, j, k) in model.N if mkt == i) 
                    == self.data.ds_dict[(i, s)]
            )
        model.demand_balance = pyo.Constraint(model.I, model.S, rule=demand_balance_rule)

        def sales_upper_rule1(model, i, j, k, s):
            return model.z[i, j, k, s] <= model.y[i, j, k]
        model.sales_upper1 = pyo.Constraint(model.N, model.S, rule=sales_upper_rule1)
    
        def sales_upper_rule2(model, i, j, k, s):
            return model.z[i, j, k, s] <= model.z0[i, s] * self.data.host_attraction[i, j, k]
        model.sales_upper2 = pyo.Constraint(model.N, model.S, rule=sales_upper_rule2)

        def bigM_rule1(model, i, j, k, s):
            return model.z[i, j, k, s] >= (
                model.y[i, j, k] - self.M * (1 - model.g[i, j, k, s])
                )
        model.bigM1 = pyo.Constraint(model.N, model.S, rule=bigM_rule1)

        def bigM_rule2(model, i, j, k, s):
            return model.z[i, j, k, s] >= model.z0[i, s] - self.M * model.g[i, j, k, s]
        model.bigM2 = pyo.Constraint(model.N, model.S, rule=bigM_rule2)

        return model

    def pprint_model(self, model: pyo.ConcreteModel):
        with open(self.output_dir / f'{model.name}.txt', 'w') as f:
            model.pprint(ostream=f)

    def solve(self, sol_file: str, log_file: str, solver_name='gurobi', tee=True):
        solver = pyo.SolverFactory(solver_name)
        options = {
            'LogFile': str(self.output_dir / log_file),
            'LogToConsole': 1 if tee else 0,
            'TimeLimit': 14400  # 4 hours
        }

        start_time = time.time()
        results = solver.solve(self.model, tee=tee, options=options)
        solve_time = time.time() - start_time

        if results.solver.termination_condition == TerminationCondition.optimal:

            sol_df = pd.DataFrame.from_dict(
                {(i, j, k): self.model.y[i, j, k].value for (i, j, k) in self.model.N},
                orient='index',
                columns=['Protection_Level']
            )

            sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])
            sol_df.reset_index(inplace=True)
            sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']
            sol_df.to_csv(self.output_dir / sol_file)

            z_sol = pd.DataFrame.from_dict(
                {(i, j, k, s): self.model.z[i, j, k, s].value for (i, j, k) in self.model.N for s in self.model.S},
                orient='index',
                columns=['Sales']
            )
            z_sol.index = pd.MultiIndex.from_tuples(z_sol.index, names=['Market', 'Services', 'FareClass', 'Scenario'])
            z_sol.reset_index(inplace=True)
            z_sol.columns = ['Market', 'Services', 'FareClass', 'Scenario', 'Sales']
            z_sol.to_csv(self.output_dir / f'{self.model.name}_zsol.csv')

            g_sol = pd.DataFrame.from_dict(
                {(i, j, k, s): self.model.g[i, j, k, s].value for (i, j, k) in self.model.N for s in self.model.S},
                orient='index', 
                columns=['Binary_G']
            )
            g_sol.index = pd.MultiIndex.from_tuples(g_sol.index, names=['Market', 'Services', 'FareClass', 'Scenario'])
            g_sol.reset_index(inplace=True)
            g_sol.columns = ['Market', 'Services', 'FareClass', 'Scenario', 'Binary_G']
            g_sol.to_csv(self.output_dir / f'{self.model.name}_gsol.csv')

            pd.Series({
                'objective': pyo.value(self.model.obj),
                'solve_time': solve_time,
            }).to_csv(self.output_dir / f'{self.model.name}_obj.csv')

            print(f"Solve model {self.model.name} completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            return sol_df
        
        return None
        

class SBLPModel:
    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        self.output_dir = output_dir
        self.model = self._build_model(model_name)

        if os.path.exists(self.output_dir) is False:
            os.makedirs(self.output_dir)

    def _cap_cons_lhs(self, flt: str, y):
        if 'f2' in self.data.fares_df.columns and not self.data.fares_df['f2'].isna().all():
            # Network case: flight can appear in f1 or f2
            df_flt = self.data.fares_df.query("f1 == @flt | f2 == @flt")
        else:
            # Single leg case: flight only appears in f1
            df_flt = self.data.fares_df.query('f1 == @flt')
        
        lhs = sum(y[t.Market, t.Services, t.fare_class] for t in df_flt.itertuples())
        return lhs

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel(name=name)
        model.name = name

        # Sets
        model.I = pyo.Set(initialize=self.data.fares_df['Market'].unique().tolist())
        model.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        model.N = pyo.Set(initialize=self.data.products)

        # Variables
        model.y = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        model.y0 = pyo.Var(model.I, domain=pyo.NonNegativeReals)

        # Objective
        def objective_rule(model):
            return sum(self.data.r[i, j, k] * model.y[i, j, k] for (i, j, k) in model.N)
        model.obj = pyo.Objective(rule=objective_rule, sense=maximize)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        model.capacity_const = pyo.Constraint(model.L, rule=capacity_rule)

        def demand_balance_rule(model, i):
            return (model.y0[i] + sum(model.y[mkt, j, k] for (mkt, j, k) in model.N if i == mkt) 
                    == self.data.mean_market_demand[i])
        model.demand_balance = pyo.Constraint(model.I, rule=demand_balance_rule)

        def scale_rule(model, i, j, k):
            return model.y[i, j, k] * self.data.OA_attraction[i] <= model.y0[i] * self.data.host_attraction[i, j, k]
        model.scale_const = pyo.Constraint(model.N, rule=scale_rule)

        return model


    def solve(self, sol_file: str, log_file: str, solver_name='gurobi', tee=True):
        solver = pyo.SolverFactory(solver_name)
        options = {
            'LogFile': str(self.output_dir / log_file),
            'LogToConsole': 1 if tee else 0,
            'TimeLimit': 14400
        }

        # Start timing
        start_time = time.time()
        results = solver.solve(self.model, tee=tee, options=options)
        # Calculate elapsed time
        solve_time = time.time() - start_time

        if results.solver.termination_condition == TerminationCondition.optimal:
            sol_df = pd.DataFrame.from_dict(
                {(i, j, k): self.model.y[i, j, k].value for (i, j, k) in self.model.N},
                orient='index',
                columns=['Protection_Level']
            )

            sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])
            sol_df.reset_index(inplace=True)
            sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']
            sol_df.to_csv(self.output_dir / sol_file)

            pd.Series({
                'objective': pyo.value(self.model.obj),
                'solve_time': solve_time,
            }).to_csv(self.output_dir / f'{self.model.name}_obj.csv')

            print(f"Solve model {self.model.name} completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            return sol_df
        
        return None


class SSBLPModel:

    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        self.output_dir = output_dir
        self.model = self._build_model(model_name)

    def _cap_cons_lhs(self, flt: str, y):
        if 'f2' in self.data.fares_df.columns and not self.data.fares_df['f2'].isna().all():
            # Network case: flight can appear in f1 or f2
            df_flt = self.data.fares_df.query("f1 == @flt | f2 == @flt")
        else:
            # Single leg case: flight only appears in f1
            df_flt = self.data.fares_df.query('f1 == @flt')
        
        lhs = sum(y[t.Market, t.Services, t.fare_class] for t in df_flt.itertuples())
        return lhs

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.name = name

        model.I = pyo.Set(initialize=self.data.fares_df['Market'].unique().tolist())
        model.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        model.N = pyo.Set(initialize=self.data.products)
        model.S = pyo.Set(initialize=range(1, self.data.num_scenarios + 1))

        model.y = pyo.Var(model.N, domain=pyo.NonNegativeReals)
        model.y0 = pyo.Var(model.I, domain=pyo.NonNegativeReals)
        model.x = pyo.Var(model.N, model.S, within=pyo.NonNegativeReals)
        model.x0 = pyo.Var(model.I, model.S, within=pyo.NonNegativeReals)

        def objective_rule(model):
            return (1 / self.data.num_scenarios) * sum(self.data.r[i, j, k] * model.x[i, j, k, s] for (i, j, k) in model.N for s in model.S)
        model.obj = pyo.Objective(rule=objective_rule, sense=maximize)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        model.capacity_const = pyo.Constraint(model.L, rule=capacity_rule)   

        def y_demand_balance_rule(model, i):
            return (model.y0[i] + sum(model.y[mkt, j, k] for (mkt, j, k) in model.N if i == mkt) 
                    == self.data.mean_market_demand[i])
        model.y_demand_balance = pyo.Constraint(model.I, rule=y_demand_balance_rule)

        def y_scale_rule(model, i, j, k):
            return model.y[i, j, k] * self.data.OA_attraction[i] <= model.y0[i] * self.data.host_attraction[i, j, k]
        model.y_scale_const = pyo.Constraint(model.N, rule=y_scale_rule)

        def x_demand_balance_rule(model, i, s):
            return (model.x0[i, s] + sum(model.x[mkt, j, k, s] for (mkt, j, k) in model.N if i == mkt) 
                    == self.data.ds_dict[(i, s)])
        model.x_demand_balance = pyo.Constraint(model.I, model.S, rule=x_demand_balance_rule)

        def x_scale_rule(model, i, j, k, s):
            return model.x[i, j, k, s] * self.data.OA_attraction[i] <= model.x0[i, s] * self.data.host_attraction[i, j, k]
        model.x_scale_const = pyo.Constraint(model.N, model.S, rule=x_scale_rule)  

        def nesting_rule(model, i, j, k, s):
            fares = sorted([chr(ord("A") + r) for r in range(8)])
            nest_cts = (
                sum(model.x[i, j, flc, s] for flc in fares if ord(flc)>=ord(k))
                <= sum(model.y[i, j, flc] for flc in fares if ord(flc)>=ord(k))
                )
            return nest_cts
        model.nesting_ct = pyo.Constraint(model.N, model.S, rule=nesting_rule)

        # model.write(str(self.output_dir / f'{model.name}.lp'), io_options={'symbolic_solver_labels': True})

        return model

    def solve(self, sol_file: str, log_file: str, solver_name='gurobi', tee=True):
        solver = pyo.SolverFactory(solver_name)
        options = {
            'LogToConsole': 1 if tee else 0,
            'TimeLimit': 14400,
            'LogFile': str(self.output_dir / log_file),
        }

        # Start timing
        start_time = time.time()
        results = solver.solve(self.model, tee=tee, options=options)
        # Calculate elapsed time
        solve_time = time.time() - start_time

        if results.solver.termination_condition == TerminationCondition.optimal:
            sol_df = pd.DataFrame.from_dict(
                {(i, j, k): self.model.y[i, j, k].value for (i, j, k) in self.model.N},
                orient='index',
                columns=['Protection_Level']
            )

            sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])
            sol_df.reset_index(inplace=True)
            sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']
            sol_df.to_csv(self.output_dir / sol_file)

            x_sol = pd.DataFrame.from_dict(
                {(i, j, k, s): self.model.x[i, j, k, s].value for (i, j, k) in self.model.N for s in self.model.S},
                orient='index',
                columns=['Sales']
            )
            x_sol.index = pd.MultiIndex.from_tuples(x_sol.index, names=['Market', 'Services', 'FareClass', 'Scenario'])
            x_sol.reset_index(inplace=True)
            x_sol.columns = ['Market', 'Services', 'FareClass', 'Scenario', 'Sales']
            x_sol.to_csv(self.output_dir / f'{self.model.name}_xsol.csv')

            pd.Series({
                'objective': pyo.value(self.model.obj),
                'solve_time': solve_time,
            }).to_csv(self.output_dir / f'{self.model.name}_obj.csv')

            print(f"Solve model {self.model.name} completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            return sol_df
        
        return None

class NestedFluidModel:
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        data_loader: AirlineDataLoader,
        market_demand: Dict[str, float] = None,
    ):
        self.data = data_loader
        self.market_demand = market_demand if market_demand is not None else self.data.mean_market_demand
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Precompute a per-(i,j) ordered fare list to implement the nesting sums in (5f)
        self.model = self._build_model(model_name)

    def _cap_cons_lhs(self, flt: str, y):
        """Unified capacity constraint left-hand side for both single leg and network cases"""
        # Check if f2 column exists and has non-null values to determine if it's network case
        if 'f2' in self.data.fares_df.columns and not self.data.fares_df['f2'].isna().all():
            # Network case: flight can appear in f1 or f2
            df_flt = self.data.fares_df.query("f1 == @flt | f2 == @flt")
        else:
            # Single leg case: flight only appears in f1
            df_flt = self.data.fares_df.query('f1 == @flt')
        
        lhs = sum(y[t.Market, t.Services, t.fare_class] for t in df_flt.itertuples())
        return lhs

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel()
        m.name = name

        # Sets
        m.I = pyo.Set(initialize=self.data.fares_df["Market"].unique().tolist())
        m.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        m.N = pyo.Set(initialize=self.data.products)  # tuples (i,j,k)

        # Variables
        m.z = pyo.Var(m.N, domain=pyo.NonNegativeReals)       # realized sales per product
        m.z0 = pyo.Var(m.I, domain=pyo.NonNegativeReals)      # outside option market sales
        m.y = pyo.Var(m.N, domain=pyo.NonNegativeReals)   # protection levels
        m.lmbda = pyo.Var(m.I, domain=pyo.NonNegativeReals)   # lambda_i

        # Objective: (5a) maximize sum r_{i,j,k} * z_{i,j,k}
        def obj_rule(m):
            return sum(self.data.r[i, j, k] * m.z[i, j, k] for (i, j, k) in m.N)
        m.obj = pyo.Objective(rule=obj_rule, sense=maximize)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        m.capacity_const = pyo.Constraint(m.L, rule=capacity_rule)

        # (5c) market demand balance: sum_jk z_{i,j,k} + z_{i0} = D_i
        def demand_balance(m, i):
            return (m.z0[i] + sum(m.z[mkt, j, k] for (mkt, j, k) in m.N if mkt == i) 
                    == self.data.mean_market_demand[i])
        m.demand_balance = pyo.Constraint(m.I, rule=demand_balance)

        # (5d) scale with λ: z_{i,j,k} ≤ v_{i,j,k} * λ_i
        def scale_upper(m, i, j, k):
            return m.z[i, j, k] <= self.data.host_attraction[(i, j, k)] * m.lmbda[i]
        m.scale_upper = pyo.Constraint(m.N, rule=scale_upper)

        # (5e) outside option lower bound: z_{i0} ≥ v_{i0} * λ_i
        def outside_lower(m, i):
            return m.z0[i] >= self.data.OA_attraction[i] * m.lmbda[i]
        m.outside_lower = pyo.Constraint(m.I, rule=outside_lower)

        fares = sorted([chr(ord("A") + r) for r in range(8)], reverse=True)
        # (5f) nesting with protection levels y:
        # z_ijH <= y_ijH
        # z_ijG + z_ijH <= y_ijG + y_ijH
        # z_ijF + z_ijG + z_ijH <= y_ijF + y_ijG + y_ijH
        # ...
        # z_ijA + ... + z_ijH <= y_ijA + ... + y_ijH
        def nesting_rule(m, i, j, k):
            ij_products = [(mkt, svc, fc) for (mkt, svc, fc) in m.N if mkt == i and svc == j]
            ij_fares = sorted(set(fc for (_, _, fc) in ij_products), reverse=True)  # H, G, F, ..., A
            
            # For fare class k, sum all sales from k down to the highest fare class
            # (k and all higher fare classes)
            lhs = sum(m.z[i, j, fc] for fc in ij_fares if ord(fc) >= ord(k))
            rhs = sum(m.y[i, j, fc] for fc in ij_fares if ord(fc) >= ord(k))
            
            return lhs <= rhs
        m.nesting = pyo.Constraint(m.N, rule=nesting_rule)

        return m

    def solve(self, sol_file: str, log_file: str, solver_name="gurobi", tee=True):
        solver = pyo.SolverFactory(solver_name)
        options = {
            "LogFile": str(self.output_dir / log_file),
            "LogToConsole": 1 if tee else 0,
        }

        # Write LP file before solving
        lp_path = self.output_dir / (self.model.name + '.lp')
        self.model.write(str(lp_path), io_options={'symbolic_solver_labels': True})

        results = solver.solve(self.model, tee=tee, options=options)

        if results.solver.termination_condition == TerminationCondition.optimal:
            # Save protection levels to sol_file (consistent with other model classes)
            sol_df = pd.DataFrame.from_dict(
                {(i, j, k): pyo.value(self.model.y[i, j, k]) for (i, j, k) in self.model.N},
                orient='index',
                columns=['Protection_Level']
            )
            sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])
            sol_df.reset_index(inplace=True)
            sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']
            sol_df.to_csv(self.output_dir / sol_file, index=False)

            # export z, z0, lambda and objective for inspection
            z_df = (
                pd.DataFrame(
                    [(i, j, k, pyo.value(self.model.z[i, j, k])) for (i, j, k) in self.model.N],
                    columns=["Market", "Services", "FareClass", "Sales"],
                )
            )
            z_df.to_csv(self.output_dir / f"{self.model.name}_z.csv", index=False)

            z0_df = pd.DataFrame(
                [(i, pyo.value(self.model.z0[i])) for i in self.model.I],
                columns=["Market", "OutsideSales"],
            )
            z0_df.to_csv(self.output_dir / f"{self.model.name}_z0.csv", index=False)

            lam_df = pd.DataFrame(
                [(i, pyo.value(self.model.lmbda[i])) for i in self.model.I],
                columns=["Market", "Lambda"],
            )
            lam_df.to_csv(self.output_dir / f"{self.model.name}_lambda.csv", index=False)

            pd.Series({"objective": pyo.value(self.model.obj)}).to_csv(
                self.output_dir / f"{self.model.name}_obj.csv"
            )
            return {
                "objective": pyo.value(self.model.obj),
                "protection_levels_df": sol_df,
                "z_df": z_df,
                "z0_df": z0_df,
                "lambda_df": lam_df,
            }
        return None

class NestedBigM:
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        data_loader: AirlineDataLoader,
        market_demand: Dict[str, float] = None,
    ):
        self.data = data_loader
        self.market_demand = market_demand if market_demand is not None else self.data.mean_market_demand
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Calculate big-M values more safely
        # Note: 'A' is highest fare class, 'H' is lowest fare class
        self.M = {}
        for (i, j, k) in self.data.products:
            # Find next lower fare class (higher alphabetically: A->B->C->...->H)
            higher_priced = chr(ord(k) - 1) if ord(k) > ord('A') else None
            
            if higher_priced:
                ratio = (self.data.host_attraction[(i, j, k)] / self.data.host_attraction[(i, j, higher_priced)])
                self.M[(i, j, k)] = max(self.market_demand[i], ratio * self.market_demand[i])
            else:
                ratio = self.data.host_attraction[(i, j, k)] / self.data.OA_attraction[i] * self.market_demand[i]
                self.M[(i, j, k)] = max(self.market_demand[i], ratio * self.market_demand[i])

        self.model = self._build_model(model_name)

    def _cap_cons_lhs(self, flt: str, y):
        """Unified capacity constraint left-hand side for both single leg and network cases"""
        # Check if f2 column exists and has non-null values to determine if it's network case
        if 'f2' in self.data.fares_df.columns and not self.data.fares_df['f2'].isna().all():
            # Network case: flight can appear in f1 or f2
            df_flt = self.data.fares_df.query("f1 == @flt | f2 == @flt")
        else:
            # Single leg case: flight only appears in f1
            df_flt = self.data.fares_df.query('f1 == @flt')
        
        lhs = sum(y[t.Market, t.Services, t.fare_class] for t in df_flt.itertuples())
        return lhs

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel()
        m.name = name

        # Sets
        m.I = pyo.Set(initialize=self.data.fares_df["Market"].unique().tolist())
        m.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        m.N = pyo.Set(initialize=self.data.products)  # tuples (i,j,k)

        # Variables
        m.z = pyo.Var(m.N, domain=pyo.NonNegativeReals)       # realized sales per product
        m.z0 = pyo.Var(m.I, domain=pyo.NonNegativeReals)      # outside option market sales
        m.y = pyo.Var(m.N, domain=pyo.NonNegativeReals)   # protection levels
        m.g = pyo.Var(m.N, domain=pyo.Binary)     # big-M binary variables
        m.R = pyo.Var(m.N, domain=pyo.NonNegativeReals)  
  
        # Objective: (5a) maximize sum r_{i,j,k} * z_{i,j,k}
        def obj_rule(m):
            return sum(self.data.r[i, j, k] * m.z[i, j, k] for (i, j, k) in m.N)
        m.obj = pyo.Objective(rule=obj_rule, sense=maximize)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        m.capacity_const = pyo.Constraint(m.L, rule=capacity_rule)

        # market demand balance: sum_jk z_{i,j,k} + z_{i0} = D_i
        def demand_balance(m, i):
            return (m.z0[i] + sum(m.z[mkt, j, k] for (mkt, j, k) in m.N if mkt == i) 
                    == self.data.mean_market_demand[i])
        m.demand_balance = pyo.Constraint(m.I, rule=demand_balance)

        # z_{ijk} <= R_{ijk}
        def scale_upper_1(m, i, j, k):
            return m.z[i, j, k] <= m.R[i, j, k]
        m.scale_upper_1 = pyo.Constraint(m.N, rule=scale_upper_1)

        # z_{ijk} <= v_{ijk} / v_{ij,k-1} * z_{ij,k-1}
        # where k-1 means next higher fare class 
        def sales_upper_2(m, i, j, k):
            higher_class = chr(ord(k) - 1) if ord(k) > ord('A') else None 
            if higher_class:
                ratio = (self.data.host_attraction[(i, j, k)] / 
                        self.data.host_attraction[(i, j, higher_class)])
                return m.z[i, j, k] <= ratio * m.z[i, j, higher_class]
            else:
                return pyo.Constraint.Skip
        m.sales_upper2 = pyo.Constraint(m.N, rule=sales_upper_2)

        # z_ijA <= z_{i0} / v_{i0} * v_{ijA} (only for the highest fare class 'A')
        def sales_upper_3(m, i, j, k):
            if k == 'A':  # only for the highest fare class
                ratio = self.data.host_attraction[(i, j, k)] / self.data.OA_attraction[i]
                return m.z[i, j, k] <= ratio * m.z0[i]
            else:
                return pyo.Constraint.Skip
        m.sales_upper3 = pyo.Constraint(m.N, rule=sales_upper_3)

        def sales_lower_1(m, i, j, k):
            return m.z[i, j, k] >= m.R[i, j, k] - self.M[i, j, k] * (1 - m.g[i, j, k])
        m.sales_lower1 = pyo.Constraint(m.N, rule=sales_lower_1)

        # z_{ijk} >= v_{ijk} / v_{ij,k-1} * z_{ij,k-1} - M * (1 - g_{ijk})
        def sales_lower_2(m, i, j, k):
            higher_class = chr(ord(k) - 1) if ord(k) > ord('A') else None
            if higher_class:
                ratio = (self.data.host_attraction[(i, j, k)] / self.data.host_attraction[(i, j, higher_class)])
                return m.z[i, j, k] >= (ratio * m.z[i, j, higher_class] - self.M[(i, j, k)] * (1 - m.g[i, j, k]))
            else:
                return pyo.Constraint.Skip
        m.sales_lower2 = pyo.Constraint(m.N, rule=sales_lower_2)

        fares = sorted([chr(ord("A") + r) for r in range(8)], reverse=True)
        def R_def(m, i, j, k):
            # Get available fare classes for this (i,j)            
            # Sum y values for k and higher fare classes (k and classes before k alphabetically)
            y_sum = sum(m.y[i, j, fc] for fc in fares if ord(fc) >= ord(k))

            # Sum z values for higher fare classes only (classes before k alphabetically)
            z_sum = sum(m.z[i, j, fc] for fc in fares if ord(fc) > ord(k))

            return m.R[i, j, k] == y_sum - z_sum
        m.R_def = pyo.Constraint(m.N, rule=R_def)

        return m

    def solve(self, sol_file: str, log_file: str, solver_name="gurobi", tee=True):
        solver = pyo.SolverFactory(solver_name)
        options = {
            "LogFile": str(self.output_dir / log_file),
            "LogToConsole": 1 if tee else 0,
        }

        # Write LP file before solving
        lp_path = self.output_dir / (self.model.name + '.lp')
        self.model.write(str(lp_path), io_options={'symbolic_solver_labels': True})

        results = solver.solve(self.model, tee=tee, options=options)

        if results.solver.termination_condition == TerminationCondition.optimal:
            # Save protection levels to sol_file (consistent with other model classes)
            sol_df = pd.DataFrame.from_dict(
                {(i, j, k): pyo.value(self.model.y[i, j, k]) for (i, j, k) in self.model.N},
                orient='index',
                columns=['Protection_Level']
            )
            sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])
            sol_df.reset_index(inplace=True)
            sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']
            sol_df.to_csv(self.output_dir / sol_file, index=False)

            # export z, z0, lambda and objective for inspection
            z_df = (
                pd.DataFrame(
                    [(i, j, k, pyo.value(self.model.z[i, j, k])) for (i, j, k) in self.model.N],
                    columns=["Market", "Services", "FareClass", "Sales"],
                )
            )
            z_df.to_csv(self.output_dir / f"{self.model.name}_z.csv", index=False)

            z0_df = pd.DataFrame(
                [(i, pyo.value(self.model.z0[i])) for i in self.model.I],
                columns=["Market", "OutsideSales"],
            )
            z0_df.to_csv(self.output_dir / f"{self.model.name}_z0.csv", index=False)

            pd.Series({"objective": pyo.value(self.model.obj)}).to_csv(
                self.output_dir / f"{self.model.name}_obj.csv"
            )
            return {
                "objective": pyo.value(self.model.obj),
                "protection_levels_df": sol_df,
                "z_df": z_df,
                "z0_df": z0_df
            }
        return None

if __name__ == "__main__":
    # Initialize once
    sample_size = 10
    sample_id = 1
    # DATA_DIR = Path(f'../SSBLPdata')   # network case
    DATA_DIR = Path(f'../SingleMarketData')   # single market case
    data_loader = AirlineDataLoader(data_dir=DATA_DIR)

    # Load the specific sample data
    """ ds_dict = data_loader.compute_market_demand(
        filename=f'30T/{sample_size}Samples/ServicesDemand{sample_size}Samples_{sample_id}.csv') 
    print(ds_dict) """

    # Create output directory with sample info
    """ saa = 'saa'
    output_dir = DATA_DIR / f'{saa}_output_30T' / f'{sample_size}Samples'

    # Create and solve the model
    saa_model = SAAModel(model_name=f"{saa}_{sample_size}Samples_{sample_id}",
                         output_dir=output_dir,
                         data_loader=data_loader)
    
    # Solve with unique output filenames
    saa_model.solve(sol_file=f'sol_{sample_size}Samples_{sample_id}.csv', 
                    log_file=f'{saa}_{sample_size}Samples_{sample_id}.log') """
    
    """ ssblp = 'ssblp'
    output_dir = DATA_DIR / f'{ssblp}_output' / f'{sample_size}Samples'
    ssblp_model = SSBLPModel(model_name=f"{ssblp}_{sample_size}Samples_{sample_id}",
                             output_dir=output_dir,
                             data_loader=data_loader)
    # Solve with unique output filenames
    ssblp_model.solve(sol_file=f'sol_{sample_size}Samples_{sample_id}.csv',
                      log_file=f'{ssblp}_{sample_size}Samples_{sample_id}.log') """
    
    
    # 
    """ milp_model = MILPModel(model_name='MILPModel',
                           output_dir=DATA_DIR / 'MILP_output',
                           data_loader=data_loader)
    milp_model.solve(sol_file='milp_solution.csv',
                     log_file='milp_log.txt') """
    
    SBLP_model = SBLPModel(model_name='SBLPModel',
                           output_dir=DATA_DIR / 'SBLP_output',
                           data_loader=data_loader)
    SBLP_model.solve(sol_file='sblp_solution.csv',
                     log_file='sblp_log.txt')

