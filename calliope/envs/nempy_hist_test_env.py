import nempy
import gymnasium as gym

from nempy.historical_inputs import mms_db, xml_cache, loaders
from nempy import markets
from nempy.historical_inputs import loaders, mms_db, \
    xml_cache, units, demand, interconnectors, constraints
import numpy as np
from gymnasium import spaces

from typing import List

import time

# ignore FutureWarning from line 346 in nempy\spot_markert_backend\solver_interface.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class NempyHistoricalEnv(gym.Env):
    """Environment for nempy.
    Uses historical dispatch
    """

    def __init__(
            self, 
            mms_db_manager: mms_db.DBManager, 
            xml_cache_manager: xml_cache.XMLCacheManager, 
            dispatch_intervals: List[str], 
            solver_name='GUROBI'
        ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        # NOTE: DUMMY
        self.action_space = spaces.Discrete(1)
        # NOTE: DUMMY
        self.observation_space = spaces.Discrete(1)

        # Our agent bids into the energy and FCAS markets.
        # These should be 10 price and volumes bands per market (9 total markets)
        # NOTE: may need to rethink if action space is too large.
        # self.action_space = spaces.Box(
            
        # )

        # Observation space should be 
        # 1) current stored energy
        # 2) previous market demand
        # 3) previous energy+fcas prices
        # possibly
        # 4) previous bids of other participants (need to simplify somehow)
        # self.observation_space = spaces.Box(
            
        # )

        # add the mms db managers
        # TODO: change to close sqlite con
        self.mms_db_manager = mms_db_manager
        self.xml_cache_manager = xml_cache_manager

        # add the raw inputs loader for this environment
        self.raw_inputs_loader = loaders.RawInputsLoader(
            nemde_xml_cache_manager=xml_cache_manager,
            market_management_system_database=mms_db_manager
        )

        # list of 5 minute dispatch intervals
        self.dispatch_intervals = dispatch_intervals

        # track current list interval
        self.interval_trk = 0

        self.solver_name = solver_name

    def step(self, action):

        

        interval = self.dispatch_intervals[self.interval_trk]
        last_time = time.time() 

        print(f'Running interval {self.interval_trk+1}: {str(interval)}')
        self.raw_inputs_loader.set_interval(interval)
        unit_inputs = units.UnitData(self.raw_inputs_loader)
        interconnector_inputs = interconnectors.InterconnectorData(self.raw_inputs_loader)
        constraint_inputs = constraints.ConstraintData(self.raw_inputs_loader)
        demand_inputs = demand.DemandData(self.raw_inputs_loader)

        unit_info = unit_inputs.get_unit_info()

        market = markets.SpotMarket(
            market_regions=['QLD1', 'NSW1', 'VIC1','SA1', 'TAS1'],
            # market_regions=['NSW1'],

            unit_info=unit_info
        )

        # override the solver name here. must be either GUROBI or CBC
        market.solver_name = self.solver_name
    
        volume_bids, price_bids = unit_inputs.get_processed_bids()
        market.set_unit_volume_bids(volume_bids)
        market.set_unit_price_bids(price_bids)

        # Set bid in capacity limits
        unit_bid_limit = unit_inputs.get_unit_bid_availability()
        market.set_unit_bid_capacity_constraints(unit_bid_limit)
        cost = constraint_inputs.get_constraint_violation_prices()['unit_capacity']
        market.make_constraints_elastic('unit_bid_capacity', violation_cost=cost)

        # Set limits provided by the unconstrained intermittent generation
        # forecasts. Primarily for wind and solar.
        unit_uigf_limit = unit_inputs.get_unit_uigf_limits()
        market.set_unconstrained_intermitent_generation_forecast_constraint(
            unit_uigf_limit)
        cost = constraint_inputs.get_constraint_violation_prices()['uigf']
        market.make_constraints_elastic('uigf_capacity', violation_cost=cost)

        # Set unit ramp rates.
        ramp_rates = unit_inputs.get_ramp_rates_used_for_energy_dispatch(run_type="fast_start_first_run")
        market.set_unit_ramp_up_constraints(
            ramp_rates.loc[:, ['unit', 'initial_output', 'ramp_up_rate']])
        market.set_unit_ramp_down_constraints(
            ramp_rates.loc[:, ['unit', 'initial_output', 'ramp_down_rate']])
        cost = constraint_inputs.get_constraint_violation_prices()['ramp_rate']
        market.make_constraints_elastic('ramp_up', violation_cost=cost)
        market.make_constraints_elastic('ramp_down', violation_cost=cost)

        # Set unit FCAS trapezium constraints.
        unit_inputs.add_fcas_trapezium_constraints()
        cost = constraint_inputs.get_constraint_violation_prices()['fcas_max_avail']
        fcas_availability = unit_inputs.get_fcas_max_availability()
        market.set_fcas_max_availability(fcas_availability)
        market.make_constraints_elastic('fcas_max_availability', cost)
        cost = constraint_inputs.get_constraint_violation_prices()['fcas_profile']
        regulation_trapeziums = unit_inputs.get_fcas_regulation_trapeziums()
        market.set_energy_and_regulation_capacity_constraints(regulation_trapeziums)
        market.make_constraints_elastic('energy_and_regulation_capacity', cost)
        scada_ramp_down_rates = unit_inputs.get_scada_ramp_down_rates_of_lower_reg_units(run_type="fast_start_first_run")
        market.set_joint_ramping_constraints_lower_reg(scada_ramp_down_rates)
        market.make_constraints_elastic('joint_ramping_lower_reg', cost)
        scada_ramp_up_rates = unit_inputs.get_scada_ramp_up_rates_of_raise_reg_units(run_type="fast_start_first_run")
        market.set_joint_ramping_constraints_raise_reg(scada_ramp_up_rates)
        market.make_constraints_elastic('joint_ramping_raise_reg', cost)
        contingency_trapeziums = unit_inputs.get_contingency_services()
        market.set_joint_capacity_constraints(contingency_trapeziums)
        market.make_constraints_elastic('joint_capacity', cost)

        # Set interconnector definitions, limits and loss models.
        interconnectors_definitions = \
            interconnector_inputs.get_interconnector_definitions()
        loss_functions, interpolation_break_points = \
            interconnector_inputs.get_interconnector_loss_model()
        market.set_interconnectors(interconnectors_definitions)
        market.set_interconnector_losses(loss_functions,
                                        interpolation_break_points)


        # Add generic constraints and FCAS market constraints.
        fcas_requirements = constraint_inputs.get_fcas_requirements()
        market.set_fcas_requirements_constraints(fcas_requirements)

        violation_costs = constraint_inputs.get_violation_costs()
        market.make_constraints_elastic('fcas', violation_cost=violation_costs)
        generic_rhs = constraint_inputs.get_rhs_and_type_excluding_regional_fcas_constraints()
        market.set_generic_constraints(generic_rhs)
        market.make_constraints_elastic('generic', violation_cost=violation_costs)
        unit_generic_lhs = constraint_inputs.get_unit_lhs()
        market.link_units_to_generic_constraints(unit_generic_lhs)
        interconnector_generic_lhs = constraint_inputs.get_interconnector_lhs()
        market.link_interconnectors_to_generic_constraints(
            interconnector_generic_lhs)

        # Set the operational demand to be met by dispatch.
        regional_demand = demand_inputs.get_operational_demand()
        market.set_demand_constraints(regional_demand)

        # Set tiebreak constraint to equalise dispatch of equally priced bids.
        # cost = constraint_inputs.get_constraint_violation_prices()['tiebreak']
        # market.set_tie_break_constraints(cost)

        # Get unit dispatch without fast start constraints and use it to
        # make fast start unit commitment decisions.
        market.dispatch()
        dispatch = market.get_unit_dispatch()

        observation, reward, terminated, truncated, info = 0, 1, False, False, {}

        self.interval_trk +=1

        if self.interval_trk >= len(self.dispatch_intervals):
            terminated = True
        
        now = time.time() - last_time
        print(f'Took {now} seconds')
        last_time = time.time()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Effectively the real "init" of the environment.
        Reset environment for the next iteration.
        """
        observation = 0
        info = {}
        self.interval_trk = 0
        return observation, info

    def close(self):
        """
        Close after running. Here we'll just close the sqlite connection.
        """
        pass