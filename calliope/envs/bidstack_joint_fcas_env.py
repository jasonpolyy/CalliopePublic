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

from calliope.defaults import \
    MARKETS, MARKETS_REG, FCAS_SERVICES, DEMAND_MIN_HACK, DEMAND_MAX_HACK,\
          MARKET_PRICE_CAP, MARKET_PRICE_FLOOR, PRICE_MIN_HACK, PRICE_MAX_HACK

import time
import pandas as pd

# ignore FutureWarning from line 346 in nempy\spot_markert_backend\solver_interface.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

EPISODE_LEN_INTERVALS = 288
INTERVALS_PER_HOUR = 12
INV_INT_PER_HOUR = 1/INTERVALS_PER_HOUR

# just query min and max demand from history to normalise here.
# going forward would need to do something smarter
# MIN_DEMAND_HACK = 5940.51
# MAX_DEMAND_HACK = 13778.13

# sign function
def sgn(expr):
    if expr > 0:
        return 1
    if expr < 0:
        return -1
    return 0

# arbitrage indicator constants
BETA = 10
GAMMA = 0.99
TAU = 0.9

def min_max_scale_demand(demand, min=DEMAND_MIN_HACK['ENERGY'], max=DEMAND_MAX_HACK['ENERGY']):
    """Closure to normalise demand."""
    return (demand - min)/(max-min)

def min_max_scale_price(price, floor=MARKET_PRICE_FLOOR, cap=MARKET_PRICE_CAP):
    """Closure to normalise price."""
    return (price - floor)/(cap-floor)

def inv_min_max_scale_price(norm_price, floor=MARKET_PRICE_FLOOR, cap=MARKET_PRICE_CAP):
    """Closure to unnormalise price"""
    return (norm_price*(cap - floor)) + floor

def clip(value, low, high): 
    """
    Clip a value between low and high
    """
    return max(low, min(value, high)) 


random.seed(1)

class BESS():
    """
    Represent a BESS with all operational characteristics as the agent in the environment.
    """

class MeritOrderJointNEMEnv(gym.Env):
    """Environment for merit order NEM.

    Only allows quantity bids per market to simplify action space.
    Effectively acts as a price taker, just bidding in at market floor or cap depending on gen/load, and $0 for FCAS.
    """

    def __init__(
            self, 
            nem: MeritOrderNEM,
            bess_config: Config = None,
            verbose: bool = False,
            episodic: bool = True,
            init_interval: int = None
        ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.verbose = verbose
        self.nem = nem
        self.bess_config = bess_config
        self.episodic_run = episodic
        self.init_interval = init_interval

        # normalise the action space for quantities between 0 and 1. these can be rescaled to BESS spec later
        MAX_BESS_CAPACITY = bess_config.capacity / bess_config.capacity
        MAX_BESS_FCAS_RREG = bess_config.raisereg_capacity / bess_config.capacity
        MAX_BESS_FCAS_LREG = bess_config.lowerreg_capacity / bess_config.capacity
        MAX_BESS_FCAS_RFAST = bess_config.raisefast_capacity / bess_config.capacity
        MAX_BESS_FCAS_LFAST = bess_config.lowerfast_capacity / bess_config.capacity
        MAX_BESS_FCAS_RSLOW = bess_config.raiseslow_capacity / bess_config.capacity
        MAX_BESS_FCAS_LSLOW = bess_config.lowerslow_capacity / bess_config.capacity
        MAX_BESS_FCAS_RDELAY = bess_config.raisedelay_capacity / bess_config.capacity
        MAX_BESS_FCAS_LDELAY = bess_config.lowerdelay_capacity / bess_config.capacity

        # FCAS participations
        fcas_actions_low = [
            0, 0,
            0, 0, 
            0, 0,
            0, 0 
        ]
        fcas_actions_high = [
            MAX_BESS_FCAS_RREG, MAX_BESS_FCAS_LREG,
            MAX_BESS_FCAS_RFAST, MAX_BESS_FCAS_LFAST, 
            MAX_BESS_FCAS_RSLOW, MAX_BESS_FCAS_LSLOW,
            MAX_BESS_FCAS_RDELAY, MAX_BESS_FCAS_LDELAY 
        ]

        action_low = np.array([0, 0] + fcas_actions_low)
        action_high = np.array([MAX_BESS_CAPACITY, MAX_BESS_CAPACITY] + fcas_actions_high)

        self.action_space = spaces.Box(
            low=action_low,
            high=action_high
        )
        self.SOC_LOW = bess_config.capacity * bess_config.min_energy * self.bess_config.duration
        self.SOC_HIGH = bess_config.capacity * bess_config.max_energy * self.bess_config.duration
        self.energy = bess_config.capacity * self.bess_config.duration

        ENERGY_DEMAND_LOW = 0
        ENERGY_DEMAND_HIGH = 1

        FCAS_LOW = 0
        FCAS_HIGH = 1

        FCAS_PRICE_LOW = 0
        FCAS_PRICE_HIGH = 1#MARKET_PRICE_CAP

        TDAY_LOW = 0
        TDAY_HIGH = 1

        ENERGY_PRICE_LOW = 0#MARKET_PRICE_FLOOR
        ENERGY_PRICE_HIGH = 1#MARKET_PRICE_CAP

        # stupid observation spaces
        demand_req_low = [
            ENERGY_DEMAND_LOW,
            FCAS_LOW, FCAS_LOW,
            FCAS_LOW, FCAS_LOW, 
            FCAS_LOW, FCAS_LOW,
            FCAS_LOW, FCAS_LOW 
        ]

        demand_req_high = [
            ENERGY_DEMAND_HIGH,
            FCAS_HIGH, FCAS_HIGH,
            FCAS_HIGH, FCAS_HIGH, 
            FCAS_HIGH, FCAS_HIGH,
            FCAS_HIGH, FCAS_HIGH 
        ]

        prices_low = [
            ENERGY_PRICE_LOW,
            FCAS_PRICE_LOW, FCAS_PRICE_LOW,
            FCAS_PRICE_LOW, FCAS_PRICE_LOW, 
            FCAS_PRICE_LOW, FCAS_PRICE_LOW,
            FCAS_PRICE_LOW, FCAS_PRICE_LOW 
        ]

        prices_high = [
            ENERGY_PRICE_HIGH,
            FCAS_PRICE_HIGH, FCAS_PRICE_HIGH,
            FCAS_PRICE_HIGH, FCAS_PRICE_HIGH, 
            FCAS_PRICE_HIGH, FCAS_PRICE_HIGH,
            FCAS_PRICE_HIGH, FCAS_PRICE_HIGH 
        ]

        # construct observation space here
        # observation_space_low = [bess_config.min_energy, TDAY_LOW] + demand_req_low + prices_low
        # observation_space_high = [bess_config.max_energy, TDAY_HIGH] + demand_req_high + prices_high
        observation_space_low = [bess_config.min_energy,] + demand_req_low + prices_low
        observation_space_high = [bess_config.max_energy] + demand_req_high + prices_high

        observation_low = np.array(observation_space_low)
        observation_high = np.array(observation_space_high)

        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high
        )

    def step(self, action):

        self.last_action = action
        
        last_time = time.time()

        if self.verbose:
            print(f'Current interval: {self.nem.data_manager.dispatch_intervals[self.nem.timestep]}')

        # charge, discharge = action#[0]
        charge, discharge, rreg, lreg, rf, lf, rs, ls, rd, ld = action#[0]

        #q_energy_bid = q_energy*self.bess_config.capacity

        # constrain bid based on available energy
        # q_discharge = discharge*self.bess_config.capacity
        # q_charge = charge*self.bess_config.capacity

        q_energy = discharge - charge
        # reg = rreg -lreg

        p_bid = -1000 #inv_min_max_scale_price(price_bid)

        # NOTE: do cooptimisations here
        q_rreg = rreg*self.bess_config.raisereg_capacity
        q_lreg = lreg*self.bess_config.lowerreg_capacity
        q_rf = rf*self.bess_config.raisefast_capacity
        q_lf = lf*self.bess_config.lowerfast_capacity
        q_rs = rs*self.bess_config.raiseslow_capacity
        q_ls = ls*self.bess_config.lowerslow_capacity
        q_rd = rd*self.bess_config.raisedelay_capacity
        q_ld = ld*self.bess_config.lowerdelay_capacity

        # max headroom for charging and discharging
        soc_headroom_c = self.SOC_HIGH - self.curr_storage
        soc_headroom_d = self.curr_storage - self.SOC_LOW

        q_energy_bid = abs(q_energy*self.bess_config.capacity)

        # q_rreg=0
        # q_lreg=0
        # if q_energy < 0:
        #     q_energy_bid = -1*clip(q_energy_bid, 0, soc_headroom_c)
        #     q_lreg = clip(q_lreg, 0, soc_headroom_c + q_energy_bid)
        #     q_rreg = clip(q_rreg, 0, soc_headroom_d)

        # elif q_energy > 0:
        #     q_energy_bid = clip(q_energy_bid, 0, soc_headroom_d)
        #     q_lreg = clip(q_lreg, 0, soc_headroom_c)
        #     q_rreg = clip(q_rreg, 0, soc_headroom_d - q_energy_bid)

        
        # Renormalise actions to full BESS spec
        # Clip energy actions depending on BESS SoC and clip FCAS for trapezium constraints
        # Clipping format is by rearranging some constraints for maximum and minimum regulation bid from 10.1145/3632775.3661960.
        # Since we have to analytically derive the max/min amount of energy rather than from constraints,
        # we express the clipped values in terms of SoC rather than max power out, to represent what the BESS can physically do.
        if q_energy < 0:
            q_charge = abs(q_energy)
            q_energy_bid = -1*max(min(q_charge*self.bess_config.capacity, soc_headroom_c), 0)
            # q_energy_bid = clip(q_charge*self.bess_config.capacity, 0, self.SOC_HIGH - q_lreg)

            # clipping FCAS actions.
            # if we are charging, we can offer any amount of raise FCAS so long as there is sufficient discharge
            q_rreg = clip(q_rreg, 0, soc_headroom_d)
            q_rf = clip(q_rf, 0, soc_headroom_d - q_rreg)
            q_rs = clip(q_rs, 0, soc_headroom_d - q_rreg)
            q_rd = clip(q_rd, 0, soc_headroom_d - q_rreg)

            # # here q_energy_bid is negative so we just (+) it
            # # if we are charging, we can only offer the remaining headroom of lower FCAS
            q_lreg = clip(q_lreg, 0, soc_headroom_c + q_energy_bid)
            q_lf = clip(q_lf, 0, soc_headroom_c + q_energy_bid - q_lreg)
            q_ls = clip(q_ls, 0, soc_headroom_c + q_energy_bid - q_lreg)
            q_ld = clip(q_ld, 0, soc_headroom_c + q_energy_bid - q_lreg)

        elif q_energy > 0 :
            q_energy_bid = max(min(q_energy*self.bess_config.capacity, soc_headroom_d), 0)

            # # clipping FCAS actions 
            q_rreg = clip(q_rreg, 0, soc_headroom_d - q_energy_bid)
            q_rf = clip(q_rf, 0, soc_headroom_d - q_energy_bid - q_rreg)
            q_rs = clip(q_rs, 0, soc_headroom_d - q_energy_bid - q_rreg)
            q_rd = clip(q_rd, 0, soc_headroom_d - q_energy_bid - q_rreg)

            # # here q_energy_bid is negative so we just (+) it
            q_lreg = clip(q_lreg, 0, soc_headroom_c)
            q_lf = clip(q_lf, 0, soc_headroom_c - q_lreg)
            q_ls = clip(q_ls, 0, soc_headroom_c - q_lreg)
            q_ld = clip(q_ld, 0, soc_headroom_c - q_lreg)

        else:
            q_energy_bid = 0
            q_rreg = clip(q_rreg, 0, soc_headroom_d)
            q_rf = clip(q_rf, 0, soc_headroom_d - q_rreg)
            q_rs = clip(q_rs, 0, soc_headroom_d - q_rreg)
            q_rd = clip(q_rd, 0, soc_headroom_d - q_rreg)

            # # here q_energy_bid is negative so we just (+) it
            q_lreg = clip(q_lreg, 0, soc_headroom_c)
            q_lf = clip(q_lf, 0, soc_headroom_c - q_lreg)
            q_ls = clip(q_ls, 0, soc_headroom_c - q_lreg)
            q_ld = clip(q_ld, 0, soc_headroom_c - q_lreg)

        # round actions to 2 decimal places to stabilise actions
        q_energy_bid = round(q_energy_bid, 2)
        q_rreg = round(q_rreg, 2)
        q_lreg = round(q_lreg, 2)

        # bids_all = [q_energy_bid]     

          
        
        # Create the bid dict
        bess_bid_dict = {
            'ENERGY': [Bid(self.bess_config.station_name, -1000, q_energy_bid, 1)],
            'RAISE6SEC': [Bid(self.bess_config.station_name, 0.0, q_rf, 1)],
            'RAISE60SEC': [Bid(self.bess_config.station_name, 0.0, q_rs, 1)],
            'RAISE5MIN': [Bid(self.bess_config.station_name, 0.0, q_rd, 1)],
            'RAISEREG': [Bid(self.bess_config.station_name, 0.0, q_rreg, 1)],
            'LOWER6SEC': [Bid(self.bess_config.station_name, 0.0, q_lf, 1)],
            'LOWER60SEC': [Bid(self.bess_config.station_name, 0.0, q_ls, 1)],
            'LOWER5MIN': [Bid(self.bess_config.station_name, 0.0, q_ld, 1)],
            'LOWERREG': [Bid(self.bess_config.station_name, 0.0, q_lreg, 1)],
        }

        # Get all bids as a dictionary
        #bids_all = [bess_bid_dict[m].quantity for m in MARKETS]   

        # add everyones bids including agents to the NEM bidstack
        self.nem.add_bids(bess_bid_dict)

        # dispatch all markets sequentially (as they are independent)
        dispatch = self.nem.dispatch()

        MARKET_TO_CFG_CAPACITY_MAPPING = {
            'ENERGY': self.bess_config.capacity,
            'RAISEREG': self.bess_config.raisereg_capacity,
            'RAISE6SEC': self.bess_config.raisefast_capacity,
            'RAISE60SEC': self.bess_config.raiseslow_capacity,
            'RAISE5MIN': self.bess_config.raisedelay_capacity,
            'LOWERREG': self.bess_config.lowerreg_capacity,
            'LOWER6SEC': self.bess_config.lowerfast_capacity,
            'LOWER60SEC': self.bess_config.lowerslow_capacity,
            'LOWER5MIN': self.bess_config.lowerdelay_capacity,
        }

        def get_bess_dispatch(service, rescale=False):
            bess_dispatch = 0
            try:
                denom = 1
                if rescale:
                    denom = MARKET_TO_CFG_CAPACITY_MAPPING[service]
                bess_dispatch = dispatch[service]['dispatch'][self.bess_config.station_name] / denom
            except:
                bess_dispatch = 0
            return bess_dispatch

        bess_dispatched_all = [get_bess_dispatch(m) for m in MARKETS]

        # get raw dispatch numbers to calculate SoC correctly
        bess_energy = get_bess_dispatch('ENERGY')
        bess_rreg = get_bess_dispatch('RAISEREG')
        bess_lreg = get_bess_dispatch('LOWERREG')

        # SoC calculated as previous observation of SoC + load - gen including efficiencies
        bess_gen = 0
        bess_load = 0

        if bess_energy > 0:
            bess_gen = bess_energy
        if bess_energy < 0:
            bess_load = abs(bess_energy)
        
        # Reward is calculated as earned revenue from arbitrage + penalty for breaching physical limits
        eta_c, eta_d = self.bess_config.charge_efficiency, self.bess_config.discharge_efficiency
        prev_storage = self.curr_storage 

        STORAGE_LOW = self.bess_config.min_energy * self.bess_config.capacity * self.bess_config.duration
        STORAGE_HIGH = self.bess_config.max_energy * self.bess_config.capacity * self.bess_config.duration

        # Get dispatch prices
        region_price = dispatch['ENERGY']['price']

        # rescale dispatches
        bess_energy_rescale = abs(bess_energy / self.bess_config.capacity)
        bess_gen_rescale = bess_gen / self.bess_config.capacity
        bess_load_rescale = bess_load / self.bess_config.capacity
        bess_rreg_rescale = bess_rreg / self.bess_config.raisereg_capacity
        bess_lreg_rescale = bess_lreg / self.bess_config.lowerreg_capacity

        # battery can charge and discharge by reg utilisation
        bess_gen_rreg_util = bess_gen + self.bess_config.rreg_util*bess_rreg
        bess_load_lreg_util = bess_load + self.bess_config.lreg_util*bess_lreg

        delta_soc = eta_c*bess_load_lreg_util - (bess_gen_rreg_util/eta_d)
        new_storage = prev_storage + INV_INT_PER_HOUR*delta_soc

        self.curr_storage = new_storage

        bess_gen_rreg_util_rescale = bess_gen_rescale + self.bess_config.rreg_util*bess_rreg_rescale
        bess_load_lreg_util_rescale = bess_load_rescale + self.bess_config.lreg_util*bess_lreg_rescale

        arbitrage = bess_energy_rescale*region_price*((bess_gen_rreg_util_rescale/eta_d)-(eta_c*bess_load_lreg_util_rescale)) 
        # arbitrage_reg_util = region_price*((bess_rreg_rescale*self.bess_config.rreg_util/eta_d) - (eta_c*bess_lreg_rescale*self.bess_config.lreg_util))

        # update exponential moving average spot price for arbitrage stuffs
        self.exp_mov_avg_price = TAU*self.exp_mov_avg_price + (1-TAU) * region_price

        # FCAS revenue. 
        # Make sure get_bess_dispatch() has rescale=True to get rescaled dispatch
        fcas_revenue = sum(
            [dispatch[service]['price']*get_bess_dispatch(service, rescale=True) 
             for service in FCAS_SERVICES]
        )

        # Arbitrage reward shaping
        charge_indicator = sgn(self.exp_mov_avg_price - region_price)
        discharge_indicator = sgn(region_price - self.exp_mov_avg_price)
        abs_diff_scalar = abs(region_price - self.exp_mov_avg_price)

        arbitrage_indicator_reward = BETA*bess_energy_rescale*abs_diff_scalar*((discharge_indicator*(bess_gen_rescale/eta_d))+(charge_indicator*eta_c*bess_load_rescale)) 
        # arbitrage_indicator_reward = BETA*abs_diff_scalar*((discharge_indicator*bess_gen_rescale/eta_d)+(charge_indicator*eta_c*bess_load_rescale)) 

        # arbitrage_indicator_reward = bess_energy_rescale*BETA*abs_diff_scalar*((discharge_indicator*bess_gen_rreg_util_rescale/eta_d)+(charge_indicator*eta_c*bess_load_lreg_util_rescale)) 
        
        charge_promoter_prev = self.charge_promoter
        self.charge_trk +=  INV_INT_PER_HOUR*eta_c*bess_load_lreg_util
        self.charge_promoter = min(self.charge_trk - STORAGE_HIGH, 0)
        pbrs = GAMMA*self.charge_promoter - charge_promoter_prev

        # reward 
        reward = arbitrage  + fcas_revenue  + arbitrage_indicator_reward #+ pbrs # + soc_violation 
        # reward = arbitrage_indicator_reward  + fcas_revenue  #+ arbitrage_indicator_reward # + soc_violation 

        # get from MeritOrderNEM property
        tday = self.nem.curr_period_id

        # demand_req_obs = [
        #     min_max_scale_demand(dispatch[m]['demand'], min=DEMAND_MIN_HACK[m], max=DEMAND_MAX_HACK[m]) 
        #     for m in MARKETS
        # ]

        forecast_demand = self.nem.get_forecast_demand_dict()
        
        # observce the next demand forecast
        demand_req_obs = [
            min_max_scale_demand(forecast_demand[f'{m}_FC'], min=DEMAND_MIN_HACK[m], max=DEMAND_MAX_HACK[m]) 
            for m in MARKETS
        ]

        price_obs = [
            min_max_scale_price(dispatch[m]['price'], floor=PRICE_MIN_HACK[m], cap=PRICE_MAX_HACK[m]) 
            for m in MARKETS
        ]

        prices = [dispatch[m]['price'] for m in MARKETS]

        # get observation here
        observation = np.array(
            [
                new_storage / self.energy, 
                #tday / EPISODE_LEN_INTERVALS
            ] + demand_req_obs + price_obs
        ).astype(np.float32)

        truncated = False
        terminated = False
        # if new_storage < STORAGE_LOW:
        #     new_storage = STORAGE_LOW
        #     truncated = True
        #     reward = -500
        # if new_storage > STORAGE_HIGH:
        #     new_storage = STORAGE_HIGH
        #     truncated = True
        #     reward = -500

        # # punish for proposing actions that violate SOC. this is different from market dispatch
        # # discharge side: proposes actions that drains too much storage
        # q_energy_bid_unconstrained = q_energy*self.bess_config.capacity
        soc_violation = 0
        # if q_energy_bid_unconstrained > 0:
        #     discharge_violation = self.curr_storage - self.SOC_LOW - INV_INT_PER_HOUR*q_energy_bid_unconstrained
        #     if discharge_violation < 0:
        #         soc_violation = discharge_violation
        #         reward = -50
        #         truncated = True
                
        # # charge side: proposes actions that breaches available headroom
        # if q_energy_bid_unconstrained < 0:
        #     charge_violation = self.SOC_HIGH - self.curr_storage + INV_INT_PER_HOUR*q_energy_bid_unconstrained
        #     if charge_violation < 0:
        #         soc_violation = charge_violation
        #         reward = -50
        #         truncated = True
        
        self.interval_trk +=1

        # termination of the episode happens after 288 intervals are run and running episodic
        if self.interval_trk >= EPISODE_LEN_INTERVALS and self.episodic_run:
            terminated = True
        
        now = time.time() - last_time
        info = {
            'time_sec': now,
            'di': self.nem.data_manager.dispatch_intervals[self.nem.timestep],
            'action': action,
            'q_bids': bess_bid_dict,
            'dispatched_services': bess_dispatched_all,
            'prices': prices,
            'soc_prev': prev_storage,
            'soc_new': new_storage,
            'reward': reward,
            'soc_violation': soc_violation,
            'truncated': truncated
        }
        if self.verbose:
            print(f'Episode took {now} seconds')
            print(f'Action: {action}')
            print(f'Quantity bids: [{bess_bid_dict}]')
            print(f'Dispatched energy: {bess_dispatched_all}')
            print(f'Dispatch prices: {prices}')
            print(f'SoC previous: {prev_storage}, SoC new: {new_storage}')
            print(f'Reward: {reward}')
            print(f'soc_violation: {soc_violation}')


        # step the nem one timestep forward
        self.nem.step()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Effectively the real "init" of the environment.
        Reset environment for the next iteration.
        """
        #observation = 0
        super().reset(seed=seed)

        init_storage = self.bess_config.init_storage * self.bess_config.capacity * self.bess_config.duration

        self.curr_storage = init_storage

        self.interval_trk = 0
        
        self.charge_promoter = 0
        self.charge_trk=0
            
        # randomly sample a start interval. make sure at least one days worth of intervals at least (EPISODE_LEN_INTERVALS)
        self.interval = random.sample(
            range(0, len(self.nem.data_manager.dispatch_intervals)-EPISODE_LEN_INTERVALS), 
            k=1
        )[0]

        if self.init_interval is not None:
            self.interval = self.init_interval


        if self.verbose:    
            print(f'Start interval: {self.nem.data_manager.dispatch_intervals[self.interval]}')

        # Initial solve of the current interval to get demand and price
        self.nem.reset()
        self.nem.timestep = self.interval
        self.nem.add_bids()
        dispatch_init = self.nem.dispatch()

        # demand_req_obs = [
        #     min_max_scale_demand(dispatch_init[m]['demand'], min=DEMAND_MIN_HACK[m], max=DEMAND_MAX_HACK[m]) 
        #     for m in MARKETS
        # ]
        forecast_demand = self.nem.get_forecast_demand_dict()

        demand_req_obs = [
            min_max_scale_demand(forecast_demand[f'{m}_FC'], min=DEMAND_MIN_HACK[m], max=DEMAND_MAX_HACK[m]) 
            for m in MARKETS
        ]
        price_obs = [
            min_max_scale_price(dispatch_init[m]['price'], floor=PRICE_MIN_HACK[m], cap=PRICE_MAX_HACK[m]) 
            for m in MARKETS
        ]

        # set initial price to average of full period
        self.exp_mov_avg_price = 25.14 #dispatch_init['ENERGY']['price']

        observation_list = [
            self.bess_config.init_storage, 
            #self.nem.curr_period_id/EPISODE_LEN_INTERVALS
        ]+ demand_req_obs+ price_obs
        
        observation = np.array(observation_list).astype(np.float32)

        info = {}

        return observation, info

    def close(self):
        """
        Close after running.
        """
        pass