from ortools.linear_solver import pywraplp
from abc import ABC, abstractmethod
import pandas as pd
from warnings import warn
import numpy as np
from calliope import nempy_utils
import time

class AbstractOptimisationModel(ABC):
    """
    Generic optimisation model that can be solved using OR-TOOLS.
    """
    def __init__(self, config, solver_backend: str = "SCIP"):
        self.solver_backend = solver_backend
        self.status = pywraplp.Solver.NOT_SOLVED
        self.config = config

    def set_parameters(self, params):
        self.params = params

    @abstractmethod
    def _add_variables(self):
        """Add the variables to the model solver itself."""
        pass

    @abstractmethod
    def _add_constraints(self):
        """Add the constraints to the model solver itself."""
        pass

    @abstractmethod
    def _add_objective(self):
        """Add the objective function to the model solver itself."""
        pass

    def _init_solver(self):
        self.solver: pywraplp.Solver = pywraplp.Solver.CreateSolver(self.solver_backend)

    def get_unit_info(self):
        """
        Get unit information for `nempy.markets.SpotMarket` class.
        """
        unit_gen = nempy_utils.format_unit_info(
            unit=self.config.gen_duid, 
            dispatch_type='generator', 
            region=self.config.region,
            loss_factor=self.config.mlf_gen
        )
        unit_load = nempy_utils.format_unit_info(
            unit=self.config.load_duid, 
            dispatch_type='load', 
            region=self.config.region,
            loss_factor=self.config.mlf_load
        )
        df = pd.concat([unit_gen, unit_load]).reset_index(drop=True)
        return df

    @abstractmethod
    def get_bids(self):
        """
        Return volume and price bids in a way `nempy` can represent them.
        """
        pass

    @abstractmethod
    def get_bid_availability(self):
        """
        Return the max availability for all markets they participate in.
        This is used for the set_unit_bid_capacity_constraints() and set_fcas_max_availability() constraints.
        """
        pass

    def build(self):
        """
        Build the model by adding parameters, variables, constraints
        and the objective function to the solver object.
        """
        self.status = pywraplp.Solver.NOT_SOLVED
        self._init_solver()
        if self.solver_backend == 'GUROBI':
            # HACK IN GAPS
            self.solver.SetSolverSpecificParametersAsString('MIPGap=0.2 MIPFocus=3')

        self._add_variables()
        self._add_constraints()
        self._add_objective()

    def solve(self, verbose=False):
        if verbose:
            print(f"Solving with {self.solver.SolverVersion()}")
        t_solve=  time.time()
        status = self.solver.Solve()
        if verbose: print(f'Solved in {time.time()-t_solve}')
        if status == pywraplp.Solver.OPTIMAL:
            if verbose:
                print("An optimal solution has been found to the problem.")
            self.status = status
        else:
            print("The solver failed to find an optimal solution for the problem.")

    def get_all_variables_as_dataframe(self):
        """
        Method to get all variables defined in self.solver.variables into a dataframe.
        """
        variables_list = [
            (*v.name().rsplit("_", 1), v.solution_value())
            for v in self.solver.variables()
        ]
        colnames = ["VARIABLE", "TIMESTEP", "VALUE"]
        df = pd.DataFrame(variables_list, columns=colnames)
        df = df.astype({"VARIABLE": str, "TIMESTEP": int, "VALUE": float})

        var_df = df.pivot(index="TIMESTEP", columns="VARIABLE", values="VALUE").reset_index(drop=True)

        return var_df

    def to_dataframe(self):
        """
        Return model as a dataframe with outputs for all variables, indexed by SETTLEMENTDATE.
        """
        if self.status != pywraplp.Solver.OPTIMAL:
            warn(
                "Model has not found an optimal solution yet. Please solve the model again!"
            )
            empty_return = pd.DataFrame()
            return empty_return

        var_df = self.get_all_variables_as_dataframe()
        param_df = self.params.to_dataframe()

        df = pd.concat([param_df, var_df], axis=1)
        return df

    def is_solved(self):
        """
        Check if model is solved.
        """
        return self.status == pywraplp.Solver.OPTIMAL