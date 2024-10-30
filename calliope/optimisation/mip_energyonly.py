from calliope.optimisation.base import AbstractOptimisationModel
import numpy as np
from calliope.optimisation import naming


class MixedIntegerESSModelEnergyOnlyPriceTaker(AbstractOptimisationModel):
    """
    Class to represent a mixed integer program for optimal dispatch of energy storage.
    """
    # fmt: off
    def _add_objective(self):
        """
        Add the objective function here.
        """
        self.objective = self.solver.Objective()

        for t in self.params.idx:
            rop = self.params.ROP[t]
            charge = self.charge[t]
            discharge = self.discharge[t]

            self.objective.SetCoefficient(discharge, rop)
            self.objective.SetCoefficient(charge, -rop)

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
        self.charge = [
            self.solver.NumVar(0, self.config.capacity, naming.CHARGE.format(t=t)) for t in self.params.idx
        ]
        self.discharge = [
            self.solver.NumVar(0, self.config.capacity, naming.DISCHARGE.format(t=t)) for t in self.params.idx
        ]
        self.soc = [
            self.solver.NumVar(min_soc, max_soc, naming.SOC.format(t=t)) for t in self.params.idx
        ]
        self.is_charge = [
            self.solver.IntVar(0, 1, naming.IS_CHARGE.format(t=t)) for t in self.params.idx
        ]
        self.cumulative_discharge = [
            self.solver.NumVar(0, np.inf, naming.CUMULATIVE_DISCHARGE.format(t=t)) for t in self.params.idx
        ]

    def _add_constraints(self):
        """
        Add all constraints for the model by calling the related methods.
        """
        self._soc_constraint()
        self._binary_dispatch_constraint()
        self._cumulative_discharge_constraint()
        self._cycle_limit()

    # Constraint definitions
    def _soc_constraint(self):
        """
        State of charge constraint
        """
        eta_c, eta_d = self.config.charge_efficiency, self.config.discharge_efficiency
        
        # Loop through all timesteps and add constraints
        for t in self.params.idx:
            if t == 0:
                self.solver.Add(self.soc[t] == (self.charge[t]*eta_c) - (self.discharge[t]/eta_d))
            else:
                self.solver.Add(self.soc[t] == self.soc[t - 1] + ((self.charge[t]*eta_c) - (self.discharge[t]/eta_d)))

    def _binary_dispatch_constraint(self):
        """
        Binary constraint for which operation mode ESS is in.
        """
        for t in self.params.idx:
            self.solver.Add(self.charge[t] <= self.config.capacity*self.is_charge[t])
            self.solver.Add(self.discharge[t] <= self.config.capacity*(1-self.is_charge[t]))

    def _cumulative_discharge_constraint(self):
        """
        Constraint to measure cumulative discharge of ESS.
        """
        for t in self.params.idx:
            if t == 0:
                self.cumulative_discharge[t] == self.discharge[t] / self.config.discharge_efficiency
            else:
                self.solver.Add(self.cumulative_discharge[t] == self.cumulative_discharge[t-1] + self.discharge[t] / self.config.discharge_efficiency)

    def _cycle_limit(self):
        """
        Constrain the number of cycles the ESS can perform over the optimisation period
        """
        self.solver.Add(self.cumulative_discharge[-1] <= (self.config.cycle_limit*len(self.params.idx)/288))
