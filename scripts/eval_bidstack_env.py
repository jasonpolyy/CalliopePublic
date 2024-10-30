import gymnasium as gym
import sqlite3
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from calliope.envs.bidstack_energy_only_env import MeritOrderEnergyOnlyNEMEnv
from calliope.optimisation.config import load_config

from calliope.sac_constrained import SACC

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

bess_config = load_config('E:/Code/Calliope/scripts/config.yml')

print('Loading env')
env = MeritOrderEnergyOnlyNEMEnv(
    nem = nem,
    bess_config = bess_config,
    verbose=True
)


model = SAC.load('E:/Code/Calliope/models/SAC_bidstack/120000.zip', env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(288*3):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)

    # collect data and sample