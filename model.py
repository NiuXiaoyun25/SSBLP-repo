import pyomo.environ as pyo
from pyomo.environ import *
import os
import pandas as pd
from pathlib import Path
from pyomo.opt import TerminationCondition

import time

from dataloader import *


class BaseModel:
    def _cap_cons_lhs(self, flt: str, y):
        if 'f2' in self.data.fares_df.columns and not self.data.fares_df['f2'].isna().all():
            df_flt = self.data.fares_df.query("f1 == @flt | f2 == @flt")
        else:
            df_flt = self.data.fares_df.query('f1 == @flt')
        lhs = sum(y[t.Market, t.Services, t.fare_class] for t in df_flt.itertuples())
        return lhs

    def _make_output_dir(self, output_dir: Path):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

class MILPModel(BaseModel): 
    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        self._make_output_dir(output_dir)
        self._calculate_model_parameters()
        self.model = self._build_model(model_name)

    def _calculate_model_parameters(self):
        """Calculate derived model parameters"""
        self.M = max(self.data.mean_market_demand.values()) / min(self.data.host_attraction.values())

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


class SAAModel(BaseModel):
    """SAA model for choice-based demand with stockout substitution (MILP)."""
    
    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        self._calculate_model_parameters()   # big-M
        self._make_output_dir(output_dir)
        self.model = self._build_model(model_name)

    def _calculate_model_parameters(self):
        """Calculate derived model parameters"""
        self.M = max(self.data.ds_dict.values())

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
            
            # Add sales_per_attraction column: z_{ijks} / v_{ijk}
            z_sol['sales_per_attraction'] = z_sol.apply(
                lambda row: row['Sales'] / self.data.host_attraction.get((row['Market'], row['Services'], row['FareClass']), 1.0)
                if self.data.host_attraction.get((row['Market'], row['Services'], row['FareClass']), 0) != 0 else 0.0,
                axis=1
            )
            
            z_sol.to_csv(self.output_dir / f'{self.model.name}_zsol.csv', index=False)

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
        

class SBLPModel(BaseModel):
    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        self._make_output_dir(output_dir)
        self.model = self._build_model(model_name)

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


class SSBLPModel(BaseModel):

    def __init__(self, model_name: str, output_dir: Path, data_loader: AirlineDataLoader):
        self.data = data_loader
        self._make_output_dir(output_dir)
        self.model = self._build_model(model_name)

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

