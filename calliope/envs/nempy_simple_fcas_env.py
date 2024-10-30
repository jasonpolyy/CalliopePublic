import nempy
import gymnasium as gym

from calliope import nempy_utils

from nempy.historical_inputs import mms_db, xml_cache, loaders
from nempy import markets
from nempy.historical_inputs import loaders, mms_db, \
    xml_cache, units, demand, interconnectors, constraints
import numpy as np
from gymnasium import spaces

from typing import List

import datetime
import random
from calliope.optimisation.config import Config

import time
import pandas as pd

# ignore FutureWarning from line 346 in nempy\spot_markert_backend\solver_interface.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MARKET_PRICE_CAP = 16600.0
MARKET_PRICE_FLOOR = -1000.0
MIN_FCAS_PRICE = 0.0
EPISODE_LEN_INTERVALS = 288
random.seed(1)

cost = {
    'regional_demand': 2175000.0, 
    'interocnnector': 16675000.0, 
    'generic_constraint': 435000.0, 
    'ramp_rate': 16747500.0, 
    'unit_capacity': 5365000.0, 
    'energy_offer': 16457500.0, 
    'fcas_profile': 2247500.0, 
    'fcas_max_avail': 2247500.0, 
    'fcas_enablement_min': 1015000.0, 
    'fcas_enablement_max': 1015000.0, 
    'fast_start': 16385000.0, 
    'mnsp_ramp_rate': 16747500.0, 
    'msnp_offer': 16457500.0, 
    'mnsp_capacity': 5292500.0, 
    'uigf': 5582500.0, 
    'voll': 14500.0,
    'tiebreak': 1e-06
}

aemo_to_nempy_service_name={
    'ENERGY': 'energy',
    'LOWER5MIN': 'lower_5min',
    'LOWER60SEC': 'lower_60s', 
    'LOWER6SEC': 'lower_6s', 
    'LOWERREG': 'lower_reg', 
    'RAISE5MIN': 'raise_5min',
    'RAISE60SEC': 'raise_60s', 
    'RAISE6SEC': 'raise_6s', 
    'RAISEREG': 'raise_reg'
}

dispatch_type_map = {
    'GENERATOR':'generator',
    'LOAD': 'load'

}

mapping = {
    'RAISEREG': ('NSW', 'raise_reg'),
    'LOWERREG': ('NSW', 'lower_reg'),
    'RAISE6SEC': ('NSW', 'raise_6s'),
    'RAISE60SEC': ('NSW', 'raise_60s'),
    'RAISE5MIN': ('NSW', 'raise_5min'),
    'LOWER6SEC': ('NSW', 'lower_6s'),
    'LOWER60SEC': ('NSW', 'lower_60s'),
    'LOWER5MIN': ('NSW', 'lower_5min')
}


class NempySimpleFCASEnv(gym.Env):
    """Environment for nempy.
    Uses historical dispatch.

    Only allows quantity bids per market to simplify action space.
    Effectively acts as a price taker, just bidding in at market floor or cap depending on gen/load, and $0 for FCAS.
    """

    def __init__(
            self, 
            data_manager,
            start_time: datetime.datetime, 
            end_time: datetime.datetime,
            solver_name='GUROBI',
            bess_config: Config = None
        ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        
        self.bess_config = bess_config
        #self.action_space = spaces.Discrete(1)

        # Action space is 4 quantity pairs for gen energy, load energy, raise and lower contingency.
        # normalise the action space for quantities between 0 and 1. these can be rescaled to BESS spec later
        MAX_BESS_CAPACITY = bess_config.capacity / bess_config.capacity
        MAX_BESS_FCAS_RFAST = bess_config.raisefast_capacity / bess_config.capacity
        MAX_BESS_FCAS_LFAST = bess_config.lowerfast_capacity / bess_config.capacity

        action_low = np.array([0.0, 0.0, 0.0, 0.0])
        action_high = np.array([MAX_BESS_CAPACITY, MAX_BESS_CAPACITY, MAX_BESS_FCAS_RFAST, MAX_BESS_FCAS_LFAST])

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high
        )

        # NOTE: DUMMY
        #self.observation_space = spaces.Discrete(1)

        SOC_LOW = bess_config.capacity * bess_config.min_energy
        SOC_HIGH = bess_config.capacity * bess_config.max_energy
        DEMAND_LOW = -100.0
        DEMAND_HIGH = 15000

        # Observation space is current battery storage, forecasted demand, previous energy and fcas prices
        observation_low = np.array([SOC_LOW, DEMAND_LOW, MARKET_PRICE_FLOOR, MIN_FCAS_PRICE])
        observation_high = np.array([SOC_HIGH, DEMAND_HIGH, MARKET_PRICE_CAP, MARKET_PRICE_CAP])

        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high
        )

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
            

        # list of 5 minute dispatch intervals
        self.dispatch_intervals = nempy_utils.get_dispatch_intervals(start_time, end_time)

        # track current list interval
        self.interval_trk = 0

        self.solver_name = solver_name


        # Preprocessing up front to maximise time not used in dispatch
        unit_info = data_manager.unit_mapping
        unit_info.columns = ['unit', 'region', 'dispatch_type']
        unit_info['region'] = 'NSW'
        unit_info['dispatch_type'] = unit_info.apply(lambda x: dispatch_type_map[x.dispatch_type], axis=1)
        self.unit_info = unit_info

        # get bids
        volume_bids = data_manager.volume_bids.copy()
        price_bids = data_manager.price_bids.copy()

        # get fcas trapeziums from bids
        fcas_trapeziums = volume_bids.loc[volume_bids.BIDTYPE != 'ENERGY']

        fcas_trapeziums = fcas_trapeziums[
            ['INTERVAL_DATETIME','DUID','BIDTYPE','MAXAVAIL','ENABLEMENTMIN','ENABLEMENTMAX','LOWBREAKPOINT','HIGHBREAKPOINT']
        ]
        fcas_trapeziums.columns = [
            'INTERVAL_DATETIME','unit', 'service',
            'max_availability', 
            'enablement_min', 'enablement_max', 
            'low_break_point', 'high_break_point'
        ]
        fcas_trapeziums['service'] = fcas_trapeziums.apply(lambda x: aemo_to_nempy_service_name[x.service], axis=1)


        # bids to nempy format
        volume_bids = volume_bids[['INTERVAL_DATETIME','DUID', 'BIDTYPE'] + [f'BANDAVAIL{i}' for i in range(1,11)]]
        volume_bids.columns = ['INTERVAL_DATETIME', 'unit', 'service'] + [f'{i}' for i in range(1,11)]
        volume_bids['service'] = volume_bids.apply(lambda x: aemo_to_nempy_service_name[x.service], axis=1)

        self.volume_bids = volume_bids

        price_bids = price_bids[['SETTLEMENTDATE','DUID', 'BIDTYPE']+[f'PRICEBAND{i}' for i in range(1,11)]]
        price_bids.columns = ['SETTLEMENTDATE', 'unit', 'service'] + [f'{i}' for i in range(1,11)]
        price_bids['service'] = price_bids.apply(lambda x: aemo_to_nempy_service_name[x.service], axis=1)

        self.price_bids = price_bids


        # get fcas requirements
        fcas_requirements = data_manager.fcas_requirements

        self.fcas_requirements = fcas_requirements


        # Demand
        demand = data_manager.regional_demand.copy()
        demand = demand[['SETTLEMENTDATE','REGIONID', 'TOTALDEMAND']]
        demand.columns = ['SETTLEMENTDATE','region', 'demand']
        demand['region'] = 'NSW'

        self.demand = demand


    def step(self, action):

        self.last_action = action
        
        interval_time = datetime.datetime.strptime(self.interval,  '%Y/%m/%d %H:%M:%S')
        self.interval = nempy_utils.get_next_dispatch_interval(interval_time)

        #interval = self.dispatch_intervals[self.interval_trk]
        
        print(f'\n|== Running interval {self.interval_trk+1}: {str(self.interval)} ==|')
        action_str = "[Energy Gen Bid, Energy Load Bid, Raise 6s, Lower 6s]"

        print(f'Actions {action_str}: \n{action}')

        # turn action into price-quantity bids
        # simple action is of the form of np.array with 8 elements
        # 1) gen energy quantity
        # 2) load energy quantity
        # 3) raise6 energy quantity
        # 4) lower6 energy quantity
        q_gen_energy, q_load_energy, q_rf, q_lf = action

        bid_gen_e, bid_load_e = 0, 0
        q_energy = q_gen_energy - q_load_energy
        if q_energy > 0:
            bid_gen_e = q_energy
        else:
            bid_load_e = abs(q_energy)

        q_gen_energy_bid = bid_gen_e*self.bess_config.capacity
        q_load_energy_bid = bid_load_e*self.bess_config.capacity
        q_rf_bid = q_rf*self.bess_config.raisefast_capacity
        q_lf_bid = q_lf*self.bess_config.lowerfast_capacity

        print(f'Quantity bids: [{q_gen_energy_bid}, {q_load_energy_bid}, {q_rf_bid}, {q_lf_bid}]')

        last_time = time.time() 
       
        # Initialise market outside
        market = markets.SpotMarket(
            unit_info=self.unit_info,
            market_regions=['NSW'])
        market.solver = 'GUROBI'

        total_time_i = time.time()
        #di_curr = data_manager.dispatch_intervals[timestep]

        volume_bids_di = self.volume_bids.loc[self.volume_bids.INTERVAL_DATETIME == self.interval]
        volume_bids_di = volume_bids_di[volume_bids_di.columns[~volume_bids_di.columns.isin(['INTERVAL_DATETIME'])]]

        price_bids_di = self.price_bids.loc[
            self.price_bids.SETTLEMENTDATE == pd.to_datetime(self.interval.date() - datetime.timedelta(seconds=1))
        ]
        price_bids_di = price_bids_di[price_bids_di.columns[~price_bids_di.columns.isin(['SETTLEMENTDATE'])]]


        demand_di = demand.loc[demand.SETTLEMENTDATE == self.interval]
        demand_di = demand_di[demand_di.columns[~demand_di.columns.isin(['SETTLEMENTDATE'])]]


        # Create constraints that enforce the top of the FCAS trapezium.
        fcas_trapeziums_di = self.fcas_trapeziums.loc[self.fcas_trapeziums.INTERVAL_DATETIME == self.interval]
        fcas_availability = fcas_trapeziums_di.loc[:, ['unit', 'service', 'max_availability']]

        fcas_requirements_di = self.fcas_requirements.loc[self.fcas_requirements.SETTLEMENTDATE == self.interval]
        fcas_requirements_di = fcas_requirements_di[fcas_requirements_di.columns[~fcas_requirements_di.columns.isin(['SETTLEMENTDATE'])]]

        # Create new DataFrame
        data = []
        for key, (region, service) in mapping.items():
            volume = list(fcas_requirements_di[key].values)[0]
            if service in ('raise_reg', 'lower_reg'):
                set = f'{region.lower()}_regulation_requirement'
            else:
                set = f'{region.lower()}_{service}_requirement'

            data.append({
                'set': set,
                'region': region,
                'service': service,
                'volume': volume,
                'type': '>='
            })

        fcas_requirements_di = pd.DataFrame(data)

        market.set_unit_volume_bids(volume_bids_di)
        market.set_unit_price_bids(price_bids_di)
        market.set_fcas_max_availability(fcas_availability)
        market.set_fcas_requirements_constraints(fcas_requirements_di)
        market.set_demand_constraints(demand_di)

        # regulation_trapeziums = fcas_capacity_constraints(regulation_trapeziums)
        # market.make_constraintsrapeziums[fcas_trapeziums['service'].isin(['raise_reg', 'lower_reg'])]
        # market.set_energy_and_regulatio_elastic('energy_and_regulation_capacity', cost['fcas_profile'])

        # contingency_trapeziums = fcas_trapeziums[~fcas_trapeziums['service'].isin(['raise_reg', 'lower_reg'])]
        # market.set_joint_capacity_constraints(contingency_trapeziums)

        dispatch_time_i = time.time()
        market.dispatch()
        # avg_dispatch_time = avg_dispatch_time + ((time.time() - dispatch_time_i) - avg_dispatch_time)/(timestep+1)
        # avg_total_time = avg_total_time + ((time.time() - total_time_i) - avg_total_time)/(timestep+1)

        dispatch = market.get_unit_dispatch()

        all_prices = market.get_energy_prices()
        all_fcas_prices = market.get_fcas_prices()

        bess_region = self.bess_config.region
        region_price = nempy_utils.extract_region_price(all_prices, bess_region)
        rf_price = nempy_utils.extract_fcas_price(all_fcas_prices, 'raise_6s')
        lf_price = nempy_utils.extract_fcas_price(all_fcas_prices, 'lower_6s')

        print(region_price, rf_price, lf_price)

        #NOTE Update SoC for battery here based on if it got dispatched
        gen_duid, load_duid = self.bess_config.gen_duid, self.bess_config.load_duid
        #bess_gen_dispatch = dispatch.loc[(dispatch.unit.isin([gen_duid])) & (dispatch.service=='energy')]
        #bess_load_dispatch = dispatch.loc[(dispatch.unit.isin([load_duid])) & (dispatch.service=='energy')]

        bess_gen = nempy_utils.extract_unit_dispatch(dispatch, 'energy', gen_duid)
        bess_load = nempy_utils.extract_unit_dispatch(dispatch, 'energy', load_duid)
        bess_rf = nempy_utils.extract_unit_dispatch(dispatch, 'raise_6s', gen_duid)
        bess_lf = nempy_utils.extract_unit_dispatch(dispatch, 'lower_6s', load_duid)

        # SoC calculated as previous observation of SoC + load - gen including efficiencies

        print(f'Dispatched gen and load: {bess_gen}, {bess_load}')
        print(f'Dispatched FCAS r6s and l6s: {bess_rf}, {bess_lf}')

        # Reward is calculated as earned revenue from arbitrage + penalty for breaching physical limits
        INTERVALS_PER_HOUR = 12
        INV_INT_PER_HOUR = 1/INTERVALS_PER_HOUR
        eta_c, eta_d = self.bess_config.charge_efficiency, self.bess_config.discharge_efficiency
        new_storage = self.curr_storage + INV_INT_PER_HOUR*((eta_c*bess_load) - (bess_gen/eta_d))
        print(f'SoC previous: {self.curr_storage}, SoC new: {new_storage}')

        self.curr_storage = new_storage

        STORAGE_LOW = self.bess_config.min_energy * self.bess_config.capacity
        STORAGE_HIGH = self.bess_config.max_energy * self.bess_config.capacity

        # Revenue from energy arbitrage
        arbitrage = region_price*(bess_gen-bess_load) 
        fcas_payment = rf_price * bess_rf + lf_price *bess_lf

        # terminate if over
        reward = arbitrage + fcas_payment
        terminated = False
        if new_storage < STORAGE_LOW:
            new_storage = STORAGE_LOW
            terminated = True
            reward = -50
        if new_storage > STORAGE_HIGH:
            new_storage = STORAGE_HIGH
            terminated = True
            reward = -50

        curr_region_demand = nempy_utils.extract_region_demand(self.regional_demand, bess_region)

        observation = np.array([new_storage, curr_region_demand, region_price, rf_price]).astype(np.float32)
        
        # TODO figure out what these gymnasium config thingys do
        truncated = False
        info =  {}

        self.interval_trk +=1

        if self.interval_trk >= EPISODE_LEN_INTERVALS:
            terminated = True
        
        now = time.time() - last_time
        print(f'Episode took {now} seconds')
        last_time = time.time()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Effectively the real "init" of the environment.
        Reset environment for the next iteration.
        """
        #observation = 0

        init_storage = self.bess_config.init_storage * self.bess_config.capacity
        init_price = 0.0
        init_demand = 0.0

        observation = np.array([init_storage, init_demand, init_price, init_price]).astype(np.float32)

        self.curr_storage = init_storage

        info = {}
        self.interval_trk = 0

        # randomly sample a start interval. make sure at least one days worth of intervals at least (EPISODE_LEN_INTERVALS)
        self.interval = random.sample(self.dispatch_intervals[:-EPISODE_LEN_INTERVALS], 1)[0]

        print(f'Reset interval: {self.interval}')

        return observation, info

    def close(self):
        """
        Close after running. Here we'll just close the sqlite connection.
        """
        pass