from calliope.optimisation.base import AbstractOptimisationModel
import numpy as np
from calliope.optimisation import naming
from warnings import warn

class MixedIntegerESSModelJointPriceTaker(AbstractOptimisationModel):
    """
    Class to represent a mixed integer program for optimal dispatch of energy storage.
    Is a co-optimised dispatch model for energy+FCAS.
    """
    # fmt: off
    def get_bids(self):
        """
        Generate the volume and price bids for all services.

        In a price taker model, a bid here is formed as (optimised MW, input price).
        We can expand these to be profiled based on some criteria.
        """  
        raise NotImplementedError

    def get_bid_availability(self):
        raise NotImplementedError
        
    def _add_objective(self):
        """
        Add the objective function here.
        """
        self.objective = self.solver.Objective()

        for t in self.params.idx:
            # energy arbitrage (without reg utilisation)
            self.objective.SetCoefficient(self.discharge[t], self.params.ROP[t])
            self.objective.SetCoefficient(self.charge[t] , -self.params.ROP[t])

            if self.params.AGC is not None:
                self.objective.SetCoefficient(self.rreg[t], self.params.ROP[t]*self._raise_reg_signal(t))
                self.objective.SetCoefficient(self.lreg[t], -self.params.ROP[t]*self._lower_reg_signal(t))

            # capacity payments
            self.objective.SetCoefficient(self.rf[t], self.params.RAISE6SECROP[t])
            self.objective.SetCoefficient(self.rs[t], self.params.RAISE60SECROP[t])
            self.objective.SetCoefficient(self.rd[t], self.params.RAISE5MINROP[t])
            self.objective.SetCoefficient(self.rreg[t], self.params.RAISEREGROP[t])
            self.objective.SetCoefficient(self.lf[t], self.params.LOWER6SECROP[t])
            self.objective.SetCoefficient(self.ls[t], self.params.LOWER60SECROP[t])
            self.objective.SetCoefficient(self.ld[t], self.params.LOWER5MINROP[t])
            self.objective.SetCoefficient(self.lreg[t], self.params.LOWERREGROP[t])


        self.objective.SetMaximization()

    def _add_variables(self):
        """
        Add parameters to the MIP model. These parameters are generally a Prices object
        """
        energy = self.config.capacity * self.config.duration
        min_soc, max_soc = list(
            np.array([self.config.min_energy, self.config.max_energy]) * energy
        )

        # Add variables for every time index
        self.charge = [self.solver.NumVar(0, self.config.capacity, naming.CHARGE.format(t=t)) for t in self.params.idx]
        self.discharge = [self.solver.NumVar(0, self.config.capacity, naming.DISCHARGE.format(t=t)) for t in self.params.idx]
        self.soc = [self.solver.NumVar(min_soc, max_soc, naming.SOC.format(t=t)) for t in self.params.idx]
        self.is_charge = [self.solver.IntVar(0, 1, naming.IS_CHARGE.format(t=t)) for t in self.params.idx]
        self.is_discharge = [self.solver.IntVar(0, 1, naming.IS_DISCHARGE.format(t=t)) for t in self.params.idx]
        self.cumulative_discharge = [self.solver.NumVar(0, np.inf, naming.CUMULATIVE_DISCHARGE.format(t=t)) for t in self.params.idx]

        # All fcas services: Raise and Lower contingency and regulation
        self.rreg = [self.solver.NumVar(0, self.config.raisereg_capacity, naming.RAISEREGULATION.format(t=t)) for t in self.params.idx]
        self.lreg = [self.solver.NumVar(0, self.config.lowerreg_capacity, naming.LOWERREGULATION.format(t=t)) for t in self.params.idx]
        self.rf = [self.solver.NumVar(0, self.config.raisefast_capacity, naming.RAISEFAST.format(t=t)) for t in self.params.idx]
        self.rs = [self.solver.NumVar(0, self.config.raiseslow_capacity, naming.RAISESLOW.format(t=t)) for t in self.params.idx]
        self.rd = [self.solver.NumVar(0, self.config.raisedelay_capacity, naming.RAISEDELAY.format(t=t)) for t in self.params.idx]
        self.lf = [self.solver.NumVar(0, self.config.lowerfast_capacity, naming.LOWERFAST.format(t=t)) for t in self.params.idx]
        self.ls = [self.solver.NumVar(0, self.config.lowerslow_capacity, naming.LOWERSLOW.format(t=t)) for t in self.params.idx]
        self.ld = [self.solver.NumVar(0, self.config.lowerdelay_capacity, naming.LOWERDELAY.format(t=t)) for t in self.params.idx]

        self.charge_reg = [self.solver.NumVar(0, self.config.capacity, naming.CHARGE_REG.format(t=t)) for t in self.params.idx]
        self.discharge_reg = [self.solver.NumVar(0, self.config.capacity, naming.DISCHARGE_REG.format(t=t)) for t in self.params.idx]

    def _add_constraints(self):
        """
        Add all constraints for the model by calling the related methods.
        """
        self._soc_constraint()
        self._binary_dispatch_constraint()
        self._cumulative_discharge_constraint()
        self._cycle_limit()
        self._trapezium_constraints()
        self._sustain_constraints()
        self._reg_utilisation_helpers()

        if self.config.final_storage is not None:
            self._final_energy_constraint()

    def _reg_utilisation_helpers(self):
        for t in self.params.idx:
            self.solver.Add(self.charge_reg[t] == self.lreg[t] * self._lower_reg_signal(t))
            self.solver.Add(self.discharge_reg[t] ==  self.rreg[t] * self._raise_reg_signal(t))

    # Constraint definitions
    def _final_energy_constraint(self):
        """
        Set the final storage numbers.
        """
        last_t = np.max(self.params.idx)
        energy = self.config.capacity * self.config.duration
        
        self.solver.Add(self.soc[last_t] == self.config.final_storage * energy)

    def _lower_reg_signal(self, t):
        """
        Calculate charge at time t considering regulation when AGC signals exist.
        """
        if self.params.AGC is None:
            return float(0)
        
        # Get charge with % of regulation service called on for utilisation
        agc_signal = self.params.AGC[t]
        return -min(agc_signal, 0.0)
            
    def _raise_reg_signal(self, t):
        """
        Calculate discharge at time t considering regulation when AGC signals exist.
        """
        if self.params.AGC is None:
            return float(0)
        
        # Get discharge with % of regulation service called on for utilisation
        agc_signal = self.params.AGC[t]
        return max(agc_signal, 0.0)
             
    def _soc_constraint(self):
        """
        State of charge constraint
        """
        eta_c, eta_d = self.config.charge_efficiency, self.config.discharge_efficiency
        intervals = self.params.intervals_per_hour
        energy = self.config.capacity * self.config.duration

        # Loop through all timesteps and add constraints
        for t in self.params.idx:
            if t == 0:
                self.solver.Add(self.soc[0] == self.config.init_storage * energy)
            else:
                # calculate the charge/discharge accounting for regulation utilisation
                charge_t = self.charge[t] + self.lreg[t] * self._lower_reg_signal(t)
                discharge_t = self.discharge[t] + self.rreg[t] * self._raise_reg_signal(t)

                rule = (self.soc[t] - self.soc[t-1] == (1/intervals)*((charge_t*eta_c) - (discharge_t/eta_d)))
                self.solver.Add(rule)

    def _binary_dispatch_constraint(self):
        """
        Binary constraint for which operation mode ESS is in.
        """
        for t in self.params.idx:
            self.solver.Add(self.charge[t] <= self.config.capacity*self.is_charge[t])
            self.solver.Add(self.discharge[t] <= self.config.capacity*self.is_discharge[t])

            # cannot charge and discharge at the same time
            self.solver.Add(self.is_charge[t] + self.is_discharge[t] <= 1)

    def _cumulative_discharge_constraint(self):
        """
        Constraint to measure cumulative discharge of ESS.
        """
        eta_d = self.config.discharge_efficiency

        for t in self.params.idx:
            if t == 0:
                self.cumulative_discharge[t] == self.discharge[t] / eta_d
            else:
                discharge_t = self.discharge[t] + self.rreg[t] * self._raise_reg_signal(t)
                self.solver.Add(self.cumulative_discharge[t] == self.cumulative_discharge[t-1] + discharge_t / eta_d)

    def _daily_cycle_limit(self):
        """
        Limit the maximum number of cycles a BESS can perform in a given day.
        """

    def _cycle_limit(self):
        """
        Constrain the number of cycles the ESS can perform over the optimisation period
        """
        intervals_per_day = 24*self.params.intervals_per_hour

        self.solver.Add(self.cumulative_discharge[-1] <= (self.config.cycle_limit*len(self.params.idx)/intervals_per_day))

    def _trapezium_constraints(self):
        """
        Add FCAS trapezium constraints that must be followed.
        These refer to constraints in FCAS MODEL IN NEMDE paper.
        https://aemo.com.au/-/media/files/electricity/nem/security_and_reliability/dispatch/policy_and_process/fcas-model-in-nemde.pdf?la=en
        """
        for t in self.params.idx:
            self.solver.Add(self.rf[t] + self.rreg[t] + self.discharge[t] <= self.config.capacity) 
            self.solver.Add(self.rs[t] + self.rreg[t] + self.discharge[t] <= self.config.capacity) 
            self.solver.Add(self.rd[t] + self.rreg[t] + self.discharge[t] <= self.config.capacity) 
            self.solver.Add(self.lf[t] + self.lreg[t] + self.charge[t] <= self.config.capacity) 
            self.solver.Add(self.ls[t] + self.lreg[t] + self.charge[t] <= self.config.capacity) 
            self.solver.Add(self.ld[t] + self.lreg[t] + self.charge[t] <= self.config.capacity) 

    def _sustain_constraints(self):
        """
        Add FCAS sustain constraints that describe the length of time active power response must
        be sustained for in case of a frequency event.
        """
        energy = self.config.capacity * self.config.duration
        min_soc, max_soc = list(
            np.array([self.config.min_energy, self.config.max_energy]) * energy
        )
        for t in self.params.idx:
            charge_t = self.charge[t] + self.lreg[t] * self._lower_reg_signal(t)
            discharge_t = self.discharge[t] + self.rreg[t] * self._raise_reg_signal(t)
            self.solver.Add(self.soc[t] + charge_t/12 + self.lf[t]/60 + self.ls[t]/12 + self.ld[t]/6 <= max_soc)
            self.solver.Add(self.soc[t] - discharge_t/12 + self.rf[t]/60 + self.rs[t]/12 + self.rd[t]/6 >= min_soc)


