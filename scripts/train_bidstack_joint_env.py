import gymnasium as gym
import sqlite3
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# from calliope.envs.bidstack_energy_only_env import MeritOrderEnergyOnlyNEMEnv
# from calliope.envs.bidstack_joint_reg_env import MeritOrderJointNEMEnv
from calliope.envs.bidstack_joint_fcas_env import MeritOrderJointNEMEnv

from calliope.optimisation.config import load_config

from calliope.sac_constrained import SACC

from calliope.data import DataManager
from calliope.market import MeritOrderNEM
import numpy as np


import datetime

con = sqlite3.connect('E:/Code/Calliope/notebooks/historical_mms.db')
start_date = '2019/01/01 00:00:00'
# end_date = '2019/01/03 00:00:00'
end_date = '2019/07/01 00:00:00'
region = 'NSW1'


print('Loading data')
data_manager = DataManager(con)
data_manager.load_data(start_date, end_date, region, nem_solve_data=True, verbose=True)
nem = MeritOrderNEM(data_manager)
nem.build()

nem.load_bids_from_pickle(
    [f'E:/Code/Calliope/notebooks/bid_dict_{i}_nsw.pkl' for i in ['jan', 'feb', 'mar', 'apr','may','jun']],
    verbose=True
)

bess_config = load_config('E:/Code/Calliope/scripts/config_pricemaker.yml')

print('Loading env')
env = MeritOrderJointNEMEnv(
    nem = nem,
    bess_config = bess_config,
    verbose=True
)

# Where do you want the logs to be
logdir = 'logs'
# What is the name of your evaluation?
tb_log_name = 'SAC_JOINT_JantoJune_PM_all_longtrain_arb_indicator_noreg'

# Load the soft actor critic algorithm.
# Here i'm loading a direct copy of the stable baselines version I planned
# to implement constrained actions on, but ended up abandoning.
model = SACC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    batch_size=512,
    tensorboard_log=logdir,
    tau=0.01,
    learning_starts=288*4,
    learning_rate=0.0001,
    #train_freq=(1, "episode"),
    seed=1,
    target_update_interval=288
)

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix=tb_log_name,
  save_replay_buffer=True,
  save_vecnormalize=True,
)

TIMESTEPS=500000
model.learn(
    total_timesteps=TIMESTEPS, 
    log_interval=4, 
    tb_log_name=tb_log_name, 
    reset_num_timesteps=False, 
    callback=[checkpoint_callback]
)
