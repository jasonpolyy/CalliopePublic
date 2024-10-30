import gurobipy as gpy
import numpy as np
from calliope.optimisation import naming
from warnings import warn
import time
import pandas as pd

class MIPJointBESSGurobi():
    """
    Represent a BESS that jointly optimises in energy and FCAS.
    Written in gurobi to have better access to paramters compared to Google OR Tools.
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def get_all_variables_as_dataframe(self):
        """
        Method to get all variables defined in self.solver.variables into a dataframe.
        """
        variables_list = [
            (v.VarName.split("[")[0], int(v.VarName.split("[")[1].split(']')[0]), v.x)
            for v in self.model.getVars()
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
        var_df = self.get_all_variables_as_dataframe()
        param_df = self.params.to_dataframe()

        df = pd.concat([param_df, var_df], axis=1)
        return df
        
    def set_parameters(self, params):
        self.params = params

    def build(self):
        env = gpy.Env(empty=True)
        env.setParam('OutputFlag',0)
        env.setParam('LogToConsole', 0)
        env.start()
        self.model = gpy.Model('BESS_model', env=env)

        self._add_variables()
        self._add_constraints()
        self._add_objective()

    def solve(self, verbose=False):
        t_solve = time.time()
        self.model.Params.Presolve =1
        self.model.Params.MIPGap=0.2
        self.model.Params.MIPFocus=2
        self.model.Params.OutputFlag=0
        
        if self.model is not None:
            self.model.optimize()
            if verbose: print(f'Solved in {time.time() - t_solve}')
    
   
    def warm_start(previous_solution):
        """
        Warm start the current model with the previous solution, moving time index by 1

        Parameters
        ----------
        previous_solution : Dict
            the previous solution
        """
        raise NotImplementedError

    def _add_objective(self):

        dch_objective = gpy.quicksum([self.discharge[t]*self.params.intervals_per_hour*self.params.ROP[t] for t in self.params.idx])
        ch_objective = gpy.quicksum([-1*self.charge[t]*self.params.intervals_per_hour*self.params.ROP[t] for t in self.params.idx])

        dch_rreg_objective = gpy.quicksum([self.rreg[t]*self.params.intervals_per_hour*self.params.ROP[t]*self._raise_reg_signal(t) for t in self.params.idx])
        ch_lreg_objective = gpy.quicksum([-1*self.lreg[t]*self.params.intervals_per_hour*self.params.ROP[t]*self._lower_reg_signal(t) for t in self.params.idx])

        rf_objective = gpy.quicksum([self.rf[t]*self.params.intervals_per_hour*self.params.RAISE6SECROP[t] for t in self.params.idx])
        rs_objective = gpy.quicksum([self.rs[t]*self.params.intervals_per_hour*self.params.RAISE60SECROP[t] for t in self.params.idx])
        rd_objective = gpy.quicksum([self.rd[t]*self.params.intervals_per_hour*self.params.RAISE5MINROP[t] for t in self.params.idx])
        rreg_objective = gpy.quicksum([self.rreg[t]*self.params.intervals_per_hour*self.params.RAISEREGROP[t] for t in self.params.idx])
        lf_objective = gpy.quicksum([self.lf[t]*self.params.intervals_per_hour*self.params.LOWER6SECROP[t] for t in self.params.idx])
        ls_objective = gpy.quicksum([self.ls[t]*self.params.intervals_per_hour*self.params.LOWER60SECROP[t] for t in self.params.idx])
        ld_objective = gpy.quicksum([self.ld[t]*self.params.intervals_per_hour*self.params.LOWER5MINROP[t] for t in self.params.idx])
        lreg_objective = gpy.quicksum([self.lreg[t]*self.params.intervals_per_hour*self.params.LOWERREGROP[t] for t in self.params.idx])

        objective = gpy.quicksum(
            [dch_objective, ch_objective, dch_rreg_objective, ch_lreg_objective,
             rf_objective, rs_objective, rd_objective, rreg_objective,
             lf_objective, ls_objective, ld_objective, lreg_objective]
        )
        self.model.setObjective(objective, gpy.GRB.MAXIMIZE)

    def _add_variables(self):
        energy = self.config.capacity * self.config.duration
        min_soc, max_soc = list(
            np.array([self.config.min_energy, self.config.max_energy]) * energy
        )

        # Add variables for every time index
        self.charge = self.model.addVars(self.params.idx, lb = 0, ub=self.config.capacity, vtype=gpy.GRB.CONTINUOUS, name='CHARGE')
        self.discharge = self.model.addVars(self.params.idx, lb = 0, ub=self.config.capacity, vtype=gpy.GRB.CONTINUOUS, name='DISCHARGE')
        self.soc = self.model.addVars(self.params.idx, lb = min_soc, ub=max_soc, vtype=gpy.GRB.CONTINUOUS, name='SOC')
        self.is_charge = self.model.addVars(self.params.idx, vtype=gpy.GRB.BINARY, name='IS_CHARGE')

        self.rreg = self.model.addVars(self.params.idx, lb = 0, ub=self.config.raisereg_capacity, vtype=gpy.GRB.CONTINUOUS, name='RAISEREGULATION')
        self.lreg = self.model.addVars(self.params.idx, lb = 0, ub=self.config.lowerreg_capacity, vtype=gpy.GRB.CONTINUOUS, name='LOWERREGULATION')
        self.rf = self.model.addVars(self.params.idx, lb = 0, ub=self.config.raisefast_capacity, vtype=gpy.GRB.CONTINUOUS, name='RAISEFAST')
        self.rs = self.model.addVars(self.params.idx, lb = 0, ub=self.config.raiseslow_capacity, vtype=gpy.GRB.CONTINUOUS, name='RAISESLOW')
        self.rd = self.model.addVars(self.params.idx, lb = 0, ub=self.config.raisedelay_capacity, vtype=gpy.GRB.CONTINUOUS, name='RAISEDELAY')
        self.lf = self.model.addVars(self.params.idx, lb = 0, ub=self.config.lowerfast_capacity, vtype=gpy.GRB.CONTINUOUS, name='LOWERFAST')
        self.ls = self.model.addVars(self.params.idx, lb = 0, ub=self.config.lowerslow_capacity, vtype=gpy.GRB.CONTINUOUS, name='LOWERSLOW')
        self.ld = self.model.addVars(self.params.idx, lb = 0, ub=self.config.lowerdelay_capacity, vtype=gpy.GRB.CONTINUOUS, name='LOWERDELAY')
 
        self.charge_reg = self.model.addVars(self.params.idx, lb = 0, ub=self.config.capacity, vtype=gpy.GRB.CONTINUOUS, name='CHARGE_FROM_LOWERREG')
        self.discharge_reg = self.model.addVars(self.params.idx, lb = 0, ub=self.config.capacity, vtype=gpy.GRB.CONTINUOUS, name='DISCHARGE_FROM_RAISEREG')
 
    def _add_constraints(self):
        """
        Add all constraints
        """
        self._soc_constraint()
        self._binary_dispatch_constraint()
        self._trapezium_constraints()
        self._reg_utilisation_helpers()


    def _soc_constraint(self):
        eta_c, eta_d = self.config.charge_efficiency, self.config.discharge_efficiency
        intervals = self.params.intervals_per_hour
        energy = self.config.capacity * self.config.duration

        # Loop through all timesteps and add constraints
        for t in self.params.idx:
            charge_t = self.charge[t] + self.lreg[t] * self._lower_reg_signal(t)
            discharge_t = self.discharge[t] + self.rreg[t] * self._raise_reg_signal(t)
            
            if t == 0:
                # HACK: override initial storage for time sequential
                if self.init_storage_override is not None:
                    self.model.addConstr(self.soc[0] == self.init_storage_override + (1/intervals)*((charge_t*eta_c) - (discharge_t/eta_d)))
                else:
                    self.model.addConstr(self.soc[0] == (self.config.init_storage * energy) + (1/intervals)*((charge_t*eta_c) - (discharge_t/eta_d)))
            else:
                # calculate the charge/discharge accounting for regulation utilisation
                rule = (self.soc[t] - self.soc[t-1] == (1/intervals)*((charge_t*eta_c) - (discharge_t/eta_d)))
                self.model.addConstr(rule)

    def _binary_dispatch_constraint(self):
        """
        Binary constraint for which operation mode ESS is in.
        """
        for t in self.params.idx:
            self.model.addConstr(self.charge[t] <= self.config.capacity*self.is_charge[t])
            self.model.addConstr(self.discharge[t] <= self.config.capacity*(1-self.is_charge[t]))

    def _reg_utilisation_helpers(self):
        for t in self.params.idx:
            self.model.addConstr(self.charge_reg[t] == self.lreg[t] * self._lower_reg_signal(t))
            self.model.addConstr(self.discharge_reg[t] ==  self.rreg[t] * self._raise_reg_signal(t))


    def _trapezium_constraints(self):
        """
        Add FCAS trapezium constraints that must be followed.
        These refer to constraints in FCAS MODEL IN NEMDE paper.
        https://aemo.com.au/-/media/files/electricity/nem/security_and_reliability/dispatch/policy_and_process/fcas-model-in-nemde.pdf?la=en
        """
        for t in self.params.idx:
            self.model.addConstr(self.rf[t] + self.rreg[t] + self.discharge[t] <= self.config.capacity) 
            self.model.addConstr(self.rs[t] + self.rreg[t] + self.discharge[t] <= self.config.capacity) 
            self.model.addConstr(self.rd[t] + self.rreg[t] + self.discharge[t] <= self.config.capacity) 
            self.model.addConstr(self.lf[t] + self.lreg[t] + self.charge[t] <= self.config.capacity) 
            self.model.addConstr(self.ls[t] + self.lreg[t] + self.charge[t] <= self.config.capacity) 
            self.model.addConstr(self.ld[t] + self.lreg[t] + self.charge[t] <= self.config.capacity) 


    def _lower_reg_signal(self, t):
        """
        Calculate charge at time t considering regulation when AGC signals exist.
        Use regulation utilisation simplification.
        """
        # if self.params.AGC is None:
        #     return float(0)
        
        # # Get charge with % of regulation service called on for utilisation
        # agc_signal = self.params.AGC[t]
        return self.config.lreg_util #-min(agc_signal, 0.0)
            
    def _raise_reg_signal(self, t):
        """
        Calculate discharge at time t considering regulation when AGC signals exist.
        Use regulation utilisation simplification.
        """
        # if self.params.AGC is None:
        #     return float(0)
        
        # # Get discharge with % of regulation service called on for utilisation
        # agc_signal = self.params.AGC[t]
        return self.config.rreg_util #max(agc_signal, 0.0) 
