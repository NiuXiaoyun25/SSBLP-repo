from collections import namedtuple
import pyomo.environ as pyo
from pyomo.environ import *
import time

import os
import pandas as pd
from dataloader import *
from model import BaseModel, add_cross_scenario_monotonicity

class NestedLP(BaseModel):
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        data_loader: AirlineDataLoader,
    ):
        self.data = data_loader
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        self.model = self._build_model(model_name)

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
        m.y = pyo.Var(m.N, domain=pyo.NonNegativeReals)  
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

        # market demand balance: sum_jk z_{i,j,k} + z_{i0} = D_i
        def demand_balance(m, i):
            return (m.z0[i] + sum(m.z[mkt, j, k] for (mkt, j, k) in m.N if mkt == i) 
                    == self.data.mean_market_demand[i])
        m.demand_balance = pyo.Constraint(m.I, rule=demand_balance)

        # scale with λ: z_{i,j,k} ≤ v_{i,j,k} * λ_i
        def scale_upper(m, i, j, k):
            return m.z[i, j, k] <= self.data.host_attraction[(i, j, k)] * m.lmbda[i]
        m.scale_upper = pyo.Constraint(m.N, rule=scale_upper)

        # outside option lower bound: z_{i0} ≥ v_{i0} * λ_i
        def outside_lower(m, i):
            return m.z0[i] >= self.data.OA_attraction[i] * m.lmbda[i]
        m.outside_lower = pyo.Constraint(m.I, rule=outside_lower)

        # Use dynamic fare classes available in m.N for each (i,j)
        def nesting_rule(m, i, j, k):
            ij_fares = sorted({fc for (mkt, svc, fc) in m.N if mkt == i and svc == j}, reverse=True)
            # For fare class k, sum all sales from k up to the highest fare class
            lhs = sum(m.z[i, j, fc] for fc in ij_fares if ord(fc) >= ord(k))
            rhs = sum(m.y[i, j, fc] for fc in ij_fares if ord(fc) >= ord(k))
            return lhs <= rhs
        m.nesting = pyo.Constraint(m.N, rule=nesting_rule)

        return m

    def _export_solution(self, sol_file: str):
        """Export solution to CSV files and return solution dictionary."""
        # Save protection levels to sol_file
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
        z_df = pd.DataFrame(
            [(i, j, k, pyo.value(self.model.z[i, j, k]), 
              pyo.value(self.model.z[i, j, k]) / self.data.host_attraction[(i, j, k)]) 
              for (i, j, k) in self.model.N],
            columns=["Market", "Services", "FareClass", "Sales", "z/v"],
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

    def solve(self, sol_file: str, log_file: str, solver_name="gurobi", tee=True, time_limit=None):
        print(f'Solving NestedLP model: {self.model.name}')
        solver = pyo.SolverFactory(solver_name)
        options = {
            "LogFile": str(self.output_dir / log_file),
            "LogToConsole": 1 if tee else 0,
        }
        if time_limit:
            options["TimeLimit"] = time_limit

        # Write LP file before solving
        lp_path = self.output_dir / (self.model.name + '.lp')
        self.model.write(str(lp_path), io_options={'symbolic_solver_labels': True})

        start_time = time.time()
        results = solver.solve(self.model, tee=tee, options=options)
        runtime = time.time() - start_time

        # Record runtime and gap
        stats = {
            "runtime": runtime,
            "gap": 0.0, 
            "termination_condition": str(results.solver.termination_condition),
            "solver_status": str(results.solver.status),
        }
        pd.DataFrame([stats]).to_csv(self.output_dir / f"{self.model.name}_stats.csv", index=False)

        if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.maxTimeLimit]:
            solution = self._export_solution(sol_file)
            solution.update(stats)
            return solution
        return None


class NestedBigM(BaseModel):
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        data_loader: AirlineDataLoader,
    ):
        self.data = data_loader
        self._make_output_dir(output_dir)
        self.M = self._compute_big_m()  # moved Big-M calculation to a method

        self.model = self._build_model(model_name)

    def _compute_big_m(self):
        # Note: 'A' is highest fare class, 'H' is lowest fare class
        M = {}
        for (i, j, k) in self.data.products:
            higher_priced = chr(ord(k) - 1) if ord(k) > ord('A') else None

            if higher_priced:
                ratio = (self.data.host_attraction[(i, j, k)] / self.data.host_attraction[(i, j, higher_priced)])
                M[(i, j, k)] = max(self.data.mean_market_demand[i], ratio * self.data.mean_market_demand[i])
            else:
                ratio = self.data.host_attraction[(i, j, k)] / self.data.OA_attraction[i] * self.data.mean_market_demand[i]
                M[(i, j, k)] = max(self.data.mean_market_demand[i], ratio * self.data.mean_market_demand[i])
        
        return M

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel()
        m.name = name

        # Sets
        m.I = pyo.Set(initialize=self.data.fares_df["Market"].unique().tolist())
        m.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        m.N = pyo.Set(initialize=self.data.products)  # tuples (i, j, k)

        # Variables
        m.z = pyo.Var(m.N, domain=pyo.NonNegativeReals)       # realized sales per product
        m.z0 = pyo.Var(m.I, domain=pyo.NonNegativeReals)      # outside option market sales
        m.y = pyo.Var(m.N, domain=pyo.NonNegativeReals)
        m.g = pyo.Var(m.N, domain=pyo.Binary)     # big-M binary variables
        m.R = pyo.Var(m.N, domain=pyo.NonNegativeReals)  
  
        # Objective: maximize sum r_{i,j,k} * z_{i,j,k}
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
        def sales_bound_by_cap(m, i, j, k):
            return m.z[i, j, k] <= m.R[i, j, k]
        m.sales_ub_1 = pyo.Constraint(m.N, rule=sales_bound_by_cap)

        def sales_nesting_bound(m, i, j, k):
            higher_class = chr(ord(k) - 1) if ord(k) > ord('A') else None 
            if higher_class:
                # z_{ijk} <= v_{ijk} / v_{ij,k-1} * z_{ij,k-1}
                ratio = self.data.host_attraction[(i, j, k)] / self.data.host_attraction[(i, j, higher_class)]
                return m.z[i, j, k] <= ratio * m.z[i, j, higher_class]
            elif k == 'A':  # highest fare class
                # z_ijA <= (v_{ijA} / v_{i0}) * z_{i0}
                ratio = self.data.host_attraction[(i, j, k)] / self.data.OA_attraction[i]
                return m.z[i, j, k] <= ratio * m.z0[i]
            else:
                return pyo.Constraint.Skip
        m.sales_ub_2 = pyo.Constraint(m.N, rule=sales_nesting_bound)

        def sales_lower_1(m, i, j, k):
            return m.z[i, j, k] >= m.R[i, j, k] - self.M[i, j, k] * m.g[i, j, k]
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

        def R_def(m, i, j, k):
            fares = sorted({fc for (mkt, svc, fc) in m.N if mkt == i and svc == j}, reverse=True)
            y_sum = sum(m.y[i, j, fc] for fc in fares if ord(fc) >= ord(k))
            z_sum = sum(m.z[i, j, fc] for fc in fares if ord(fc) > ord(k))

            return m.R[i, j, k] == y_sum - z_sum
        m.R_def = pyo.Constraint(m.N, rule=R_def)

        return m

    def _export_solution(self, sol_file: str):
        """Export solution to CSV files and return solution dictionary."""
        # Save protection levels to sol_file
        sol_df = pd.DataFrame.from_dict(
            {(i, j, k): pyo.value(self.model.y[i, j, k]) for (i, j, k) in self.model.N},
            orient='index',
            columns=['Protection_Level']
        )
        sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])
        sol_df.reset_index(inplace=True)
        sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']
        sol_df.to_csv(self.output_dir / sol_file, index=False)

        # export z, z0 and objective for inspection
        z_df = pd.DataFrame(
            [(i, j, k, pyo.value(self.model.z[i, j, k]), 
              pyo.value(self.model.z[i, j, k]) / self.data.host_attraction[(i, j, k)]) 
             for (i, j, k) in self.model.N],
            columns=["Market", "Services", "FareClass", "Sales", "z/v"],
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
        }

    def solve(self, sol_file: str, log_file: str, solver_name="gurobi", tee=True, time_limit=None):
        print(f'Solving NestedBigM model: {self.model.name}')
        solver = pyo.SolverFactory(solver_name)
        options = {
            "LogFile": str(self.output_dir / log_file),
            "LogToConsole": 1 if tee else 0,
        }
        if time_limit:
            options["TimeLimit"] = time_limit

        # Write LP file before solving
        lp_path = self.output_dir / (self.model.name + '.lp')
        self.model.write(str(lp_path), io_options={'symbolic_solver_labels': True})

        start_time = time.time()
        results = solver.solve(self.model, tee=tee, options=options)
        runtime = time.time() - start_time

        # Extract gap for MIP
        gap = 0.0
        if hasattr(results.problem, 'upper_bound') and hasattr(results.problem, 'lower_bound'):
            ub = results.problem.upper_bound
            lb = results.problem.lower_bound
            if ub and lb and abs(lb) > 1e-10:
                gap = abs(ub - lb) / abs(lb)

        stats = {
            "runtime": runtime,
            "gap": gap,
            "termination_condition": str(results.solver.termination_condition),
            "solver_status": str(results.solver.status),
        }
        pd.DataFrame([stats]).to_csv(self.output_dir / f"{self.model.name}_stats.csv", index=False)

        if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.maxTimeLimit]:
            solution = self._export_solution(sol_file)
            solution.update(stats)
            return solution
        return None


class SampleBasedNestedBigM(BaseModel):
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        data_loader: AirlineDataLoader,
        sample_size: int,
        sample_id: int,
    ):
        self.data = data_loader
        self.sample_size = sample_size
        self.sample_id = sample_id
        self.bigM_vals = self._compute_big_m()  
        self._make_output_dir(output_dir) 
        self.model = self._build_model(model_name)

    def _compute_big_m(self):
        rmg_cap_M = {}
        sales_bound_M = {}

        D_i = {i: max(self.data.ds_dict[i, s] for s in range(1, self.sample_size + 1))
               for i in self.data.fares_df["Market"].unique().tolist()}
        for row in self.data.fares_df.itertuples():
            bottleneck_cap = getattr(row, 'max_seats')
            rmg_cap_M[(row.Market, row.Services, row.fare_class)] = max(bottleneck_cap, D_i[row.Market])

        for i, j, k in self.data.products:
            sales_bound_M[(i, j, k)] = self.data.host_attraction[(i, j, k)] / self.data.OA_attraction[i] * D_i[i]
        
        BigMValues = namedtuple('BigMValues', ['rmg_cap_M', 'sales_bound_M'])

        return BigMValues(rmg_cap_M, sales_bound_M)
    
    def _build_model(self, name: str) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel()
        m.name = name

        # Sets
        m.I = pyo.Set(initialize=self.data.fares_df["Market"].unique().tolist())
        m.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        m.N = pyo.Set(initialize=self.data.products)  # tuples (i, j, k)
        m.S = pyo.Set(initialize=range(1, self.sample_size + 1))  # demand samples

        # Variables
        m.z = pyo.Var(m.N, m.S, domain=pyo.NonNegativeReals)       # realized sales per product
        m.z0 = pyo.Var(m.I, m.S, domain=pyo.NonNegativeReals)      # outside option market sales
        m.y = pyo.Var(m.N, domain=pyo.NonNegativeReals)   # protection levels
        m.g = pyo.Var(m.N, m.S, domain=pyo.Binary)     # big-M binary variables
        m.R = pyo.Var(m.N, m.S, domain=pyo.NonNegativeReals)

        # Objective: maximize sum r_{i,j,k} * z_{i,j,k}
        def obj_rule(m):
            return 1/self.sample_size * sum(self.data.r[i, j, k] * m.z[i, j, k, s] for (i, j, k) in m.N for s in m.S)
        m.obj = pyo.Objective(rule=obj_rule, sense=maximize)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        m.capacity_const = pyo.Constraint(m.L, rule=capacity_rule)       
        
        # Demand balance
        def demand_balance_rule(model, i, s):
            return (
                    model.z0[i, s] + sum(model.z[mkt, j, k, s] for (mkt, j, k) in model.N if mkt == i) 
                    == self.data.ds_dict[(i, s)]
                    )
        m.demand_balance = pyo.Constraint(m.I, m.S, rule=demand_balance_rule)

        # z_{ijk}^{s} <= R_{ijk}^{s}
        def sales_bound_by_cap(m, i, j, k, s):
            return m.z[i, j, k, s] <= m.R[i, j, k, s]
        m.sales_ub_1 = pyo.Constraint(m.N, m.S, rule=sales_bound_by_cap)

        def sales_nesting_bound(m, i, j, k, s):
            higher_class = chr(ord(k) - 1) if ord(k) > ord('A') else None 
            if higher_class:
                # z_{ijk}^{s} <= (v_{ijk} / v_{ij,k-1}) * z_{ij,k-1}^{s}
                ratio = (self.data.host_attraction[(i, j, k)] / self.data.host_attraction[(i, j, higher_class)])
                return m.z[i, j, k, s] <= ratio * m.z[i, j, higher_class, s]
            else:
                ratio = self.data.host_attraction[(i, j, k)] / self.data.OA_attraction[i]
                return m.z[i, j, k, s] <= ratio * m.z0[i, s]
        m.sales_ub_2 = pyo.Constraint(m.N, m.S, rule=sales_nesting_bound)

        def sales_lower_1(m, i, j, k, s):
            return m.z[i, j, k, s] >= m.R[i, j, k, s] - self.bigM_vals.rmg_cap_M[i, j, k] * m.g[i, j, k, s]
        m.sales_lower1 = pyo.Constraint(m.N, m.S, rule=sales_lower_1)

        # z_{ijk} >= v_{ijk} / v_{ij,k-1} * z_{ij,k-1} - M * (1 - g_{ijk})
        def sales_lower_2(m, i, j, k, s):
            higher_class = chr(ord(k) - 1) if ord(k) > ord('A') else None
            if higher_class:
                ratio = (self.data.host_attraction[(i, j, k)] / self.data.host_attraction[(i, j, higher_class)])
                return m.z[i, j, k, s] >= (ratio * m.z[i, j, higher_class, s] 
                                           - self.bigM_vals.sales_bound_M[(i, j, k)] * (1 - m.g[i, j, k, s]))
            else:
                return pyo.Constraint.Skip
        m.sales_lower2 = pyo.Constraint(m.N, m.S, rule=sales_lower_2)

        def R_def(m, i, j, k, s):
            fares = sorted([chr(ord("A") + r) for r in range(8)], reverse=True)
            y_sum = sum(m.y[i, j, fc] for fc in fares if ord(fc) >= ord(k))
            z_sum = sum(m.z[i, j, fc, s] for fc in fares if ord(fc) > ord(k))

            return m.R[i, j, k, s] == y_sum - z_sum
        m.R_def = pyo.Constraint(m.N, m.S, rule=R_def)

        return m

    def _export_solution(self, sol_file: str):
        """Export solution to CSV files and return solution dictionary."""
        # Save protection levels to sol_file
        sol_df = pd.DataFrame.from_dict(
            {(i, j, k): pyo.value(self.model.y[i, j, k]) for (i, j, k) in self.model.N},
            orient='index',
            columns=['Protection_Level']
        )
        sol_df.index = pd.MultiIndex.from_tuples(sol_df.index, names=['Market', 'Services', 'FareClass'])
        sol_df.reset_index(inplace=True)
        sol_df.columns = ['Market', 'Services', 'FareClass', 'Protection_Level']
        sol_df.to_csv(self.output_dir / sol_file, index=False)

        # export z, z0, g and objective for inspection
        z_df = pd.DataFrame(
            [(i, j, k, s, pyo.value(self.model.z[i, j, k, s]), 
              pyo.value(self.model.z[i, j, k, s]) / self.data.host_attraction[(i, j, k)]) 
             for (i, j, k) in self.model.N for s in self.model.S],
            columns=["Market", "Services", "FareClass", "Sample", "Sales", "z/v"],
        )
        z_df.to_csv(self.output_dir / f"{self.model.name}_z.csv", index=False)

        z0_df = pd.DataFrame(
            [(i, s, pyo.value(self.model.z0[i, s]),
              pyo.value(self.model.z0[i, s]) / self.data.OA_attraction[i]) 
              for i in self.model.I for s in self.model.S],
            columns=["Market", "Sample", "OutsideSales", "z0/v"],
        )
        z0_df.to_csv(self.output_dir / f"{self.model.name}_z0.csv", index=False)

        g_df = pd.DataFrame(
            [(i, j, k, s, pyo.value(self.model.g[i, j, k, s])) for (i, j, k) in self.model.N for s in self.model.S],
            columns=["Market", "Services", "FareClass", "Sample", "BigM_Binary"],
        )
        g_df.to_csv(self.output_dir / f"{self.model.name}_g.csv", index=False)
        
        pd.Series({"objective": pyo.value(self.model.obj)}).to_csv(
            self.output_dir / f"{self.model.name}_obj.csv"
        )
        
        return {
            "objective": pyo.value(self.model.obj),
            "protection_levels_df": sol_df,
            "z_df": z_df,
            "z0_df": z0_df
        }

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
            return self._export_solution(sol_file)
        return None


class SampleBasedNestedLP(BaseModel):
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        data_loader: AirlineDataLoader,
        sample_size: int,
        sample_id: int,
    ):
        self.data = data_loader
        self.sample_size = sample_size
        self.sample_id = sample_id
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        self.model = self._build_model(model_name)

    def _build_model(self, name: str) -> pyo.ConcreteModel:
        m = pyo.ConcreteModel()
        m.name = name

        # Sets
        m.I = pyo.Set(initialize=self.data.fares_df["Market"].unique().tolist())
        m.L = pyo.Set(initialize=self.data.flights.index.unique().tolist())
        m.N = pyo.Set(initialize=self.data.products)  # tuples (i,j,k)
        m.S = pyo.Set(initialize=range(1, self.sample_size + 1))  

        # Variables
        m.z = pyo.Var(m.N, m.S, domain=pyo.NonNegativeReals)       # realized sales per product
        m.z0 = pyo.Var(m.I, m.S, domain=pyo.NonNegativeReals)      # outside option market sales
        m.y = pyo.Var(m.N, domain=pyo.NonNegativeReals)   # protection levels
        m.lmbda = pyo.Var(m.I, m.S, domain=pyo.NonNegativeReals)   

        # Objective: maximize sum r_{i,j,k} * z_{i,j,k}
        def obj_rule(m):
            return 1 / self.sample_size * sum(self.data.r[i, j, k] * m.z[i, j, k, s] for (i, j, k) in m.N for s in m.S)
        m.obj = pyo.Objective(rule=obj_rule, sense=maximize)

        # capacity constraints.
        # \sum_i \sum_j \sum_k A_{ijkl} y_{ijk} \le C_l
        # matrix A is not available explicitly
        def capacity_rule(model, l):
            return (self._cap_cons_lhs(flt=l, y=model.y) <= self.data.flights.loc[l, 'CapRmng'])
        m.capacity_const = pyo.Constraint(m.L, rule=capacity_rule)

        # market demand balance: sum_jk z_{i,j,k}^{s} + z_{i0}^{s} = D_i^{(s)}
        def demand_balance(m, i, s):
            return (m.z0[i, s] + sum(m.z[mkt, j, k, s] for (mkt, j, k) in m.N if mkt == i) 
                    == self.data.ds_dict[(i, s)])
        m.demand_balance = pyo.Constraint(m.I, m.S, rule=demand_balance)

        # scale with λ: z_{i,j,k}^{s} ≤ v_{i,j,k} * λ_{i}^{s}
        def scale_upper(m, i, j, k, s):
            return m.z[i, j, k, s] <= self.data.host_attraction[(i, j, k)] * m.lmbda[i, s]
        m.scale_upper = pyo.Constraint(m.N, m.S, rule=scale_upper)

        # outside option lower bound: z_{i0}^{s} ≥ v_{i0} * λ_{i}^{s}
        def outside_lower(m, i, s):
            return m.z0[i, s] >= self.data.OA_attraction[i] * m.lmbda[i, s]
        m.outside_lower = pyo.Constraint(m.I, m.S, rule=outside_lower)
        
        # (5f) nesting with protection levels y:
        # z_ijH <= y_ijH
        # z_ijG + z_ijH <= y_ijG + y_ijH
        # z_ijF + z_ijG + z_ijH <= y_ijF + y_ijG + y_ijH
        # ...
        # z_ijA + ... + z_ijH <= y_ijA + ... + y_ijH
        def nesting_rule(m, i, j, k, s):
            fares = sorted([chr(ord("A") + r) for r in range(8)], reverse=True)

            lhs = sum(m.z[i, j, fc, s] for fc in fares if ord(fc) >= ord(k))
            rhs = sum(m.y[i, j, fc] for fc in fares if ord(fc) >= ord(k))

            return lhs <= rhs
        m.nesting = pyo.Constraint(m.N, m.S, rule=nesting_rule)

        return m
 
    def _export_solution(self, sol_file: str):
        """Export solution to CSV files and return solution dictionary."""
        # Save protection levels to sol_file
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
        z_df = pd.DataFrame(
            [(i, j, k, s, pyo.value(self.model.z[i, j, k, s]), 
              pyo.value(self.model.z[i, j, k, s]) / self.data.host_attraction[(i, j, k)]) 
             for (i, j, k) in self.model.N 
             for s in self.model.S],
            columns=["Market", "Services", "FareClass", "Sample", "Sales", "z/v"],
        )
        z_df.to_csv(self.output_dir / f"{self.model.name}_z.csv", index=False)

        z0_df = pd.DataFrame(
            [(i, s, pyo.value(self.model.z0[i, s])) for i in self.model.I for s in self.model.S],
            columns=["Market", "Sample", "OutsideSales"],
        )
        z0_df.to_csv(self.output_dir / f"{self.model.name}_z0.csv", index=False)

        lam_df = pd.DataFrame(
            [(i, s, pyo.value(self.model.lmbda[i, s])) for i in self.model.I for s in self.model.S],
            columns=["Market", "Sample", "Lambda"],
        )
        lam_df.to_csv(self.output_dir / f"{self.model.name}_lambda.csv", index=False)

        pd.Series({"objective": pyo.value(self.model.obj)}).to_csv(
            self.output_dir / f"{self.model.name}_obj.csv"
        )
        
        return {
            "objective": pyo.value(self.model.obj),
            "protection_levels_df": sol_df,
            "z_df": z_df,
            "z0_df": z0_df
        }

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
            return self._export_solution(sol_file)
        return None

    def solve_fixed_y(self, y_values: pd.DataFrame, solver_name="gurobi", tee=False, write_lp=False):
        """
        Solve the subproblem (MIP) for given protection levels y.

        Args:
            y_values: pd.DataFrame with columns ['Market', 'Services', 'FareClass', 'Protection_Level']
            solver_name: solver name
            tee: solver output
            write_lp: if True writes LP of subproblem
        Returns:
            dict with objective and variable solutions
        """
        sub_m = self.model.clone()
        lookup = y_values.set_index(['Market', 'Services', 'FareClass'])['Protection_Level']
        # Fix y
        for (i, j, k) in sub_m.N:
            sub_m.y[i, j, k].fix(lookup.get((i, j, k), 0.0))

        if write_lp:
            lp_path = self.output_dir / f"{sub_m.name}_fixed_y.lp"
            sub_m.write(str(lp_path), io_options={'symbolic_solver_labels': True})

        solver = pyo.SolverFactory(solver_name)
        res = solver.solve(sub_m, tee=tee)

        term = res.solver.termination_condition
        if term == TerminationCondition.optimal:
            obj = pyo.value(sub_m.obj)
            z_vals = {(i, j, k, s): pyo.value(sub_m.z[i, j, k, s]) for (i, j, k) in sub_m.N for s in sub_m.S}
            z0_vals = {(i, s): pyo.value(sub_m.z0[i, s]) for i in sub_m.I for s in sub_m.S}
            R_vals = {(i, j, k, s): pyo.value(sub_m.R[i, j, k, s]) for (i, j, k) in sub_m.N for s in sub_m.S}
            g_vals = {(i, j, k, s): pyo.value(sub_m.g[i, j, k, s]) for (i, j, k) in sub_m.N for s in sub_m.S}
            return {
                "termination": term,
                "objective": obj,
                "z": z_vals,
                "z0": z0_vals,
                "R": R_vals,
                "g": g_vals,
            }
        return {"termination": term, "objective": None}

