from stable_baselines3.common.env_checker import check_env

# from calliope.nempy_hist_test_env import NempyHistoricalEnv
# from calliope.nempy_hist_simple_env import NempyHistoricalSimpleEnv
from calliope.envs.bidstack_env import SelfScheduleMeritOrderNEMEnv
from calliope.envs.bidstack_energy_only_env import MeritOrderEnergyOnlyNEMEnv
from calliope.envs.bidstack_joint_env import MeritOrderJointNEMEnv

import sqlite3

from calliope.optimisation.config import load_config

from calliope.data import DataManager
from calliope.market import MeritOrderNEM

import datetime


con = sqlite3.connect('E:/Code/Calliope/notebooks/historical_mms.db')
start_date = '2019/01/01 00:00:00'
end_date = '2019/02/01 00:00:00'

region = 'NSW1'

print('Loading data')
data_manager = DataManager(con)
data_manager.load_data(start_date, end_date, region)
nem = MeritOrderNEM(data_manager)
nem.build()
nem.load_bids_from_pickle('E:/Code/Calliope/notebooks/bid_dict_jan_nsw.pkl')

bess_config = load_config('E:/Code/Calliope/scripts/config_pricemaker.yml')

print('Loading env')
env = MeritOrderJointNEMEnv(
    nem = nem,
    bess_config = bess_config,
    verbose=True
)

check_env(env)

