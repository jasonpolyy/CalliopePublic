import nempy
import gymnasium as gym

#from calliope.data import DataManager

from calliope.market import MeritOrderNEM, Bid


import numpy as np
from gymnasium import spaces

from typing import List

import datetime
import random
from calliope.optimisation.config import Config
from calliope.market import Bid

import time
import pandas as pd

# ignore FutureWarning from line 346 in nempy\spot_markert_backend\solver_interface.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

MARKET_PRICE_CAP = 16600.0
MARKET_PRICE_FLOOR = -1000.0
MIN_FCAS_PRICE = 0.0
EPISODE_LEN_INTERVALS = 288
INTERVALS_PER_HOUR = 12
INV_INT_PER_HOUR = 1/INTERVALS_PER_HOUR

# just query min and max demand from history to normalise here.
# going forward would need to do something smarter
MIN_DEMAND_HACK = 5940.51
MAX_DEMAND_HACK = 13778.13

# arbitrage indicator constants
BETA = 10
TAU = 0.9

def min_max_scale_demand(demand):
    """Closure to normalise demand."""
    return (demand - MIN_DEMAND_HACK)/(MAX_DEMAND_HACK-MIN_DEMAND_HACK)

def min_max_scale_price(price):
    """Closure to normalise price."""
    return (price - MARKET_PRICE_FLOOR)/(MARKET_PRICE_CAP-MARKET_PRICE_FLOOR)

def inv_min_max_scale_price(norm_price):
    """Closure to unnormalise price"""
    return (norm_price*(MARKET_PRICE_CAP - MARKET_PRICE_FLOOR)) + MARKET_PRICE_FLOOR

random.seed(1)

class BESS():
    """
    Represent a BESS with all operational characteristics as the agent in the environment.
    """

class SelfScheduleMeritOrderNEMEnv(gym.Env):
    """Environment for merit order NEM.

    Only allows quantity bids per market to simplify action space.
    Effectively acts as a price taker, just bidding in at market floor or cap depending on gen/load, and $0 for FCAS.
    """

    def __init__(
            self, 
            nem: MeritOrderNEM,
            bess_config: Config = None,
            verbose: bool = False
        ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.verbose = verbose
        self.nem = nem
        self.bess_config = bess_config

        # Action space is 4 quantity pairs for gen energy, load energy, raise and lower contingency.
        # normalise the action space for quantities between 0 and 1. these can be rescaled to BESS spec later
        MAX_BESS_CAPACITY = bess_config.capacity / bess_config.capacity
        MAX_BESS_FCAS_RFAST = bess_config.raisefast_capacity / bess_config.capacity
        MAX_BESS_FCAS_LFAST = bess_config.lowerfast_capacity / bess_config.capacity

        action_low = np.array([0, 0, 0])
        action_high = np.array([MAX_BESS_CAPACITY, MAX_BESS_CAPACITY, 1])

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high
        )

        self.SOC_LOW = bess_config.capacity * bess_config.min_energy * self.bess_config.duration
        self.SOC_HIGH = bess_config.capacity * bess_config.max_energy * self.bess_config.duration
        self.energy = bess_config.capacity * self.bess_config.duration

        DEMAND_LOW = -100.0
        DEMAND_HIGH = 15000


        FCAS_LOW = 0
        FCAS_HIGH = 1000
        TDAY_LOW = 0
        TDAY_HIGH = 1

        # Observation space is current battery storage, forecasted demand, previous energy and fcas prices
        observation_low = np.array([bess_config.min_energy, TDAY_LOW, DEMAND_LOW, MARKET_PRICE_FLOOR])
        observation_high = np.array([bess_config.max_energy, TDAY_HIGH, DEMAND_HIGH, MARKET_PRICE_CAP])

        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high
        )

    def step(self, action):

        self.last_action = action
        
        last_time = time.time()

        if self.verbose:
            print(f'Current interval: {self.nem.data_manager.dispatch_intervals[self.nem.timestep]}')

        charge, discharge, price_bid = action#[0]
        #q_energy_bid = q_energy*self.bess_config.capacity

        # constrain bid based on available energy
        q_energy = discharge - charge

        p_bid = inv_min_max_scale_price(price_bid)

        if q_energy < 0:
            q_charge = abs(q_energy)
            q_energy_bid = -1*max(min(q_charge*self.bess_config.capacity, self.SOC_HIGH - self.curr_storage), 0)#*INV_INT_PER_HOUR
        elif q_energy > 0 :
            q_energy_bid = max(min(q_energy*self.bess_config.capacity, self.curr_storage - self.SOC_LOW), 0)#*INV_INT_PER_HOUR
        else:
            q_energy_bid = 0

        # Create the bid dict
        bess_bid_dict = {
            'ENERGY': [Bid(self.bess_config.station_name, p_bid, q_energy_bid, 1)],
            'RAISE6SEC': [Bid(self.bess_config.station_name, 0, 0, 1)],
            'RAISE60SEC': [Bid(self.bess_config.station_name, 0, 0, 1)],
            'RAISE5MIN': [Bid(self.bess_config.station_name, 0, 0, 1)],
            'RAISEREG': [Bid(self.bess_config.station_name, 0, 0, 1)],
            'LOWER6SEC': [Bid(self.bess_config.station_name, 0, 0, 1)],
            'LOWER60SEC': [Bid(self.bess_config.station_name, 0, 0, 1)],
            'LOWER5MIN': [Bid(self.bess_config.station_name, 0, 0, 1)],
            'LOWERREG': [Bid(self.bess_config.station_name, 0, 0, 1)],
        }

        self.nem.add_bids(bess_bid_dict)
        dispatch = self.nem.dispatch()
        try:
            bess_energy = dispatch['ENERGY']['dispatch'][self.bess_config.station_name]
        except:
            bess_energy = 0

        # SoC calculated as previous observation of SoC + load - gen including efficiencies

        bess_gen = 0
        bess_load = 0

        if bess_energy > 0:
            bess_gen = bess_energy
            bess_load = 0
        if bess_energy < 0:
            bess_gen = 0
            bess_load = abs(bess_energy)
        
        # Reward is calculated as earned revenue from arbitrage + penalty for breaching physical limits

        eta_c, eta_d = self.bess_config.charge_efficiency, self.bess_config.discharge_efficiency
        prev_storage = self.curr_storage 
        new_storage = prev_storage + INV_INT_PER_HOUR*((eta_c*bess_load) - (bess_gen/eta_d))

        self.curr_storage = new_storage

        STORAGE_LOW = self.bess_config.min_energy * self.bess_config.capacity * self.bess_config.duration
        STORAGE_HIGH = self.bess_config.max_energy * self.bess_config.capacity * self.bess_config.duration

        # Revenue from energy arbitrage
        region_price = dispatch['ENERGY']['price']


        # rescale dispatches
        bess_gen_rescale = bess_gen / self.bess_config.capacity
        bess_load_rescale = bess_load / self.bess_config.capacity
        # bess_rf_rescale = bess_rf / self.bess_config.raisefast_capacity
        # bess_lf_rescale = bess_lf / self.bess_config.lowerfast_capacity

        arbitrage = region_price*((bess_gen_rescale/eta_d)-(eta_c*bess_load_rescale)) 

        # rf_price = dispatch['RAISE6SEC']['price']
        # lf_price = dispatch['LOWER6SEC']['price']

        # fcas_payment = (rf_price * bess_rf_rescale) + (lf_price * bess_lf_rescale)

        curr_region_demand = dispatch['ENERGY']['demand']
        # curr_rf_req = dispatch['RAISE6SEC']['demand']
        # curr_lf_req = dispatch['LOWER6SEC']['demand']

        # update exponential moving average spot price for arbitrage stuffs
        self.exp_mov_avg_price = TAU*self.exp_mov_avg_price + (1-TAU) * region_price

        def sgn(expr):
            if expr > 0:
                return 1
            if expr < 0:
                return -1
            return 0


        charge_indicator = sgn(self.exp_mov_avg_price - region_price)
        discharge_indicator = sgn(region_price - self.exp_mov_avg_price)
        arbitrage_indicator_reward = BETA*self.exp_mov_avg_price*((discharge_indicator*(bess_gen_rescale/eta_d))+(charge_indicator*eta_c*bess_load_rescale)) 

        # terminate if over
        reward = arbitrage + arbitrage_indicator_reward

        # get from MeritOrderNEM property
        tday = self.nem.curr_period_id

        # get observation here
        observation = np.array(
            [
                new_storage / self.energy, 
                tday / EPISODE_LEN_INTERVALS,
                min_max_scale_demand(curr_region_demand), 
                region_price
            ]
        ).astype(np.float32)

        truncated = False
        terminated = False
        if new_storage < STORAGE_LOW:
            new_storage = STORAGE_LOW
            truncated = True
            reward = -500
        if new_storage > STORAGE_HIGH:
            new_storage = STORAGE_HIGH
            truncated = True
            reward = -500
        
        self.interval_trk +=1

        # termination of the episode happens after 288 intervals are run
        if self.interval_trk >= EPISODE_LEN_INTERVALS:
            terminated = True
        
        now = time.time() - last_time
        info = {
            'time_sec': now,
            'action': action,
            'q_bid': q_energy_bid,
            'dispatched_energy': bess_energy,
            'energy_price': region_price,
            'soc_prev': prev_storage,
            'soc_new': new_storage,
            'reward': reward
        }
        if self.verbose:
            print(f'Episode took {now} seconds')
            print(f'Action: {action}')
            print(f'Price bids: [{p_bid}]')
            print(f'Quantity bids: [{q_energy_bid}]')
            print(f'Dispatched energy: {bess_energy}')
            # print(f'Dispatched FCAS r6s and l6s: {bess_rf}, {bess_lf}')
            print(f'Dispatch prices: {region_price}')
            print(f'SoC previous: {prev_storage}, SoC new: {new_storage}')
            print(f'Reward: {reward}')

        # step the nem one timestep forward
        self.nem.step()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Effectively the real "init" of the environment.
        Reset environment for the next iteration.
        """
        #observation = 0

        init_storage = self.bess_config.init_storage * self.bess_config.capacity * self.bess_config.duration

        self.curr_storage = init_storage

        self.interval_trk = 0
        self.exp_mov_avg_price = 0
        
        # randomly sample a start interval. make sure at least one days worth of intervals at least (EPISODE_LEN_INTERVALS)
        self.interval = random.sample(
            range(0, len(self.nem.data_manager.dispatch_intervals)-EPISODE_LEN_INTERVALS), 
            k=1
        )[0]

        if self.verbose:    
            print(f'Start interval: {self.nem.data_manager.dispatch_intervals[self.interval]}')

        # Initial solve of the current interval to get demand and price
        self.nem.reset()
        self.nem.timestep = self.interval
        self.nem.add_bids()
        dispatch_init = self.nem.dispatch()

        init_demand = dispatch_init['ENERGY']['demand']
        init_price = dispatch_init['ENERGY']['price']


        observation = np.array(
            [
                self.bess_config.init_storage, 
                self.nem.curr_period_id /EPISODE_LEN_INTERVALS,
                min_max_scale_demand(init_demand), 
                init_price
            ]
        ).astype(np.float32)

        info = {}

        return observation, info

    def close(self):
        """
        Close after running.
        """
        pass