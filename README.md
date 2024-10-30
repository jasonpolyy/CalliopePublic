# Calliope: Reinforcement learning for price-maker energy storage

<img src='res/lyre.png' height=100px>

> Calliope - the goddess of music, eloquence, and poetry. Nothing to do with energy markets, I just like the name.

Public (FINAL) Repository for jpol002 Master's Thesis on applications of deep reinforcement learning to strategic energy storage system bidding.

All core code is stored in the `calliope` module. Notebooks and scripts for experimental and batch scripting.

# Summary
The goal of this research is to study strategic joint bidding behaviour of energy storage systems in multiple markets, using deep reinforcement learning. Here the markets are modelled using simple merit order stacks in order to simplify the environment to be efficiently solved using DRL. Historical demand and bids from the National Energy Market (NEM) of Australia are used for the simulation. The key assumption here is that the battery energy storage system (BESS) is a price maker, which means it can exert market power through its actions and thus needs to learn to anticipate its impact on the market through its actions in order to maximise its individual performance.

Originally `nempy` was going to be used as the environment to fully capture the co-optimised markets of energy and FCAS in the environment. However the solve time per dispatch interval was prohibitively slow for a DRL agent to learn anything meaningful. As such a more complete version of the NEM will only be considered once merit order model evaluations have completed with success. Current tests have reduced interval solve time to 0.1 seconds including setting up the MIP problem. However this is still a super simple version of the NEM.

### Historical training period
The year 2019 is utilised for historical data, partially due to the full availability of bidding data in an SQLite database through the `historical_inputs` module of `nempy`, making querying and data management much easier than if a bespoke data management solution was built from stored CSV files. 2019 is also a somewhat average demand year in the NEM (source needed, trust me bro) and is the final year pre-COVID which should provide a standard baseline to measure performance for. 

Obviously the market is changing rapidly, and many BESS projects are in the pipeline and being commissioned in response to the decommissioning of coal units and increased renewable penetration. 2023 would also serve as a candidate year for analysis, though the recent introduction of the raise and lower 1 second markets can lead to a large distribution shift in bidding in those markets and other FCAS markets due to the nature of NEMDE co-optimisation.

# Installation (dev)
Create a conda environment using the environment.yml file.
```bash
git clone https://github.com/jasonpolyy/Calliope.git
cd Calliope
conda env create --name calliope_venv --file=environment.yml
```

The core modules to use should be installed as with any other package.
Run in editable mode to actively develop.
```bash
pip install -e .
```
If using VSCode as your IDE, you can directly use the conda env as the kernel for the notebooks.

Otherwise, you'll need to create a separate IPykernel in order to use in JupyterLab/Notebooks/IPython. See `invoke kernel` below.

# Guide to recreate results

The key components are the environment, the data and the charts. The installation above gets you the correct environment to run the code. 

**NOTE**: I've "attempted" to clean up the repo and at least explain how to get results for each part, however there's potentially areas I missed. The below will get you almost everything you need. The notebooks are a mess I'll admit so I've copy pasted the majority of the code in markdown as needed.

## Data

All data is sourced using the `nempy` package as this was the original DRL environment, and gathers all data into a local SQLite database. Make sure you're at the root of the directory `Calliope` after cloning. The following script will retrieve data from NEMWEB and parse into the database.

### Getting historical data
```bash
python scripts/nempy_db_download.py "C:/Users/YOUR_PATH" --download_inputs
```

### Data manager
The `calliope.data` module contains all the functionality to get a data manager that will be used to easily filter and select data for the experiments. The `nem_solve_data` function will only query data needed for running experiments, as bidding tables contain a large amount of data that is not necessary for loading at this stage.

```python
con = sqlite3.connect('C:/Users/YOUR_PATH/historical_mms.db')
start_date = '2019/07/01 00:00:00'
end_date = '2019/09/01 00:00:00'
region = 'NSW1'
data_manager = DataManager(con)
data_manager.load_data(start_date, end_date, region, nem_solve_data=True)
```

### Bids
The bidding data is preprocessed into a specific format using the `DataToBidObjInterface` in the `calliope.market` module. 

This was such that all historical bids could be preprocessed, and easily looked up each dispatch interval to save on data time during training. The data gets saved as pickles for each month. **NOTE**: This process takes a while and requires a lot of storage space and RAM.
```bash
python scripts/bidding_data_transform.py
```
This basically looks like this. It's a very bad script honestly.
```python
months = ['jan', 'feb', 'mar','apr', 'may', 'jun', 'jul','aug','sep','oct']
region = 'NSW1'
def tomnthstr(mnth):
    return f'0{mnth}' if mnth <10 else f'{mnth}'
for mnth, mn in zip(mnths, months):
    print(f'Running month {tomnthstr(mnth)}')
    start_date = f'2019/{tomnthstr(mnth)}/01 00:00:00'
    end_date = f'2019/{tomnthstr(mnth+1)}/01 00:00:00'
    print(f'Loading data')
    data_manager.load_data(start_date, end_date, region)
    print(f'Creating bids')
    data_to_bid = DataToBidObjInterface(data_manager)
    bid_dict = data_to_bid.data_to_bids()
    print(f'Saving to bid files')
    with open(f'bid_dict_{mn}_nsw.pkl', 'wb') as f:
        pickle.dump(bid_dict, f, pickle.HIGHEST_PROTOCOL)
    print('Done')
```
### Merit order environment

This data including the pre-processed bids can then be used to build the environment `MeritOrderNEM`. This environment is what the DRL agents will interact with and learn from.
```python
months = ['jan', 'feb', 'mar','apr', 'may', 'jun', 'jul','aug','sep','oct']
nem = MeritOrderNEM(data_manager)
nem.build()
nem.load_bids_from_pickle(
    [f'C:/Users/YOUR_PATH/bid_dict_{i}_nsw.pkl' for i in months]
)
```
## Training
The script `train_bidstack_joint_env.py` can be used to train a DRL agent as a price-taker or a price-maker, depending on which configuration is used. It's a rudimentary script with no arguments but it can be easily understood and variables changed.

This script will run for a while (500,000 timesteps) and will checkpoint a lot of intermediate results to the `logs` directory for tensorboard. If you wish to reduce this volume, you can set `save_replay_buffer=False` inside `CheckpointCallback`.

```bash
python scripts/train_bidstack_joint_env.py
```

## Evaluations
The model parameters that have been trained will be saved to `tb_log_name` as zip files every 1000 steps. This can then be read in to evaluate the model using `Evaluate.ipynb`. That notebook is basically a mess as well, but the gist of running it is like so:

```python
# same nem, bess_config and data_manager as before.
con = sqlite3.connect('C:/Users/YOUR_PATH/historical_mms.db')
start_date = '2019/07/01 00:00:00'
end_date = '2019/09/01 00:00:00'
region = 'NSW1'
print('Loading data')
data_manager = DataManager(con)
data_manager.load_data(start_date, end_date, region, nem_solve_data=True)
nem = MeritOrderNEM(data_manager)
nem.build()
# eval over july-august only
nem.load_bids_from_pickle(
    [f'C:/Users/YOUR_PATH/bid_dict_{i}_nsw.pkl' for i in ['jul', 'aug']]
)
import tqdm
bess_config = load_config('Calliope/scripts/config_pricemaker.yml')

print('Loading env')
env = MeritOrderJointNEMEnv(
    nem = nem,
    bess_config = bess_config,
    verbose=False,
    episodic=False,
    init_interval=0
)

# load trained model
model_loc = 'C:/Users/YOUR_PATH/logs/'
model_name = 'SAC_JOINT_JantoJune_PM_all_longtrain_arb_indicator_noreg_160000_steps'
model_path = f'{model_loc}{model_name}.zip'
model = SAC.load(model_path, env=env)

# options for reset()
vec_env = model.get_env()
obs = vec_env.reset()
allobs = []
allactions = []
all_prices = []
all_dispatch = []
soc_track = []
dispatch_intervals = []
all_bids = []

for i in tqdm(range(len(data_manager.dispatch_intervals))):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)

    # track all observations and actions
    allobs.append(obs)
    allactions.append(action)

    # track all prices, dispatched amounts, state of charge and timestamp
    all_prices.append(info[0]['prices'])
    all_dispatch.append(info[0]['dispatched_services'])
    soc_track.append(obs[:,0][0])
    dispatch_intervals.append(info[0]['di'])  
    all_bids.append(info[0]['q_bids'])  

env.reset()
dispatch_df = pd.DataFrame(all_dispatch, columns = [f'{m}_dispatch' for m in MARKETS])
prices_df = pd.DataFrame(all_prices, columns = [f'{m}_price' for m in MARKETS])
soc_df = pd.DataFrame(soc_track, columns = ['SOC'])
di_df = pd.DataFrame(dispatch_intervals, columns = ['SETTLEMENTDATE'])
results_df = pd.concat([di_df, dispatch_df, prices_df, soc_df], axis=1)
```
You should now be able to get the input data, train your models and evaluate them.


## Price-taker and price-maker energy-only baselines
The baselines are similarly trained and evaluated, albeit with the following changes:
* Price-taker uses `config.yml` to train, not `config_pricemaker.yml`. When evaluating, THEN use `config_pricemaker.yml` such that the learned parameters from price-taker bidding are applied to a larger battery.
* Energy-only baseline instead uses `scripts/train_bidstack_env.py`, which contains a simpler energy only version of the bidstack joint environment for the agent to train over.

## Baseline MPC
The MPC model was originally implemented using Google OR-Tools and can use any MIP solver. However now it was simplified to just using the Gurobi Python API. 

This optimisation model itself is in `calliope.optimisation.mip_joint_gurobi`. The model predictive control code is in the `mpc` module, and can be run using `mpc.ipynb.`

### LSTM price forecasts for MPC

The price forecasts for the MPC can be trained using `scripts/train_lstm_price_forecasts.py`. The training and test data is generated by running the NEM merit order model using historical bidding data in order to have a good representation of the general price space from that market model.

This can be generated using `scripts/create_merit_order_prices.py`.

```python
data_manager.load_data(start_date, end_date, region)
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug','sep','oct']
bid_dict_files=[f'C:/Users/YOUR_PATH/bid_dict_{i}_nsw.pkl' for i in months]
nem = MeritOrderNEM(data_manager)
nem.build()
nem.load_bids_from_pickle(bid_dict_files)
from calliope.market import MARKETS
prices = {m: [] for m in MARKETS}
prices['timestamp'] = []
# run thru and append market prices
i = 0
while not nem.terminated:
    i+=1
    if not i%288:
        print(f'Current: {nem.data_manager.dispatch_intervals[nem.timestep]}')
        print(f'DI {i}')
    nem.add_bids()
    dispatch = nem.dispatch()
    prices['timestamp'].append(nem.data_manager.dispatch_intervals[nem.timestep])
    nem.step()
    for m in MARKETS:
        prices[m].append(dispatch[m]['price'])
nem.reset()
# Save to csv
pd.DataFrame(prices).to_csv('merit_order_prices.csv', index=False)
```

The these will be the input prices to train an LSTM model for each market. At this point I got completely lazy and so I manually set the best models for each market inside `calliope.defaults`. Please change these when running `train_lstm_price_forecasts` and training on your side:

```python
# create multi dimensional datastructure of rolling horizon forecasts for MPC to use
BEST_MODELS = {
    'ROP': 'lstm_ROP_forecast_20240923_114607_best.pth',
    'RAISE6SECROP': 'lstm_RAISE6SECROP_forecast_20240923_122604_best.pth',
    'RAISE60SECROP': 'lstm_RAISE60SECROP_forecast_20240923_123620_best.pth',
    'RAISE5MINROP': 'lstm_RAISE5MINROP_forecast_20240923_124639_best.pth',
    'RAISEREGROP': 'lstm_RAISEREGROP_forecast_20240923_121545_best.pth',
    'LOWER6SECROP': 'lstm_LOWER6SECROP_forecast_20240923_130614_best.pth',
    'LOWER60SECROP': 'lstm_LOWER60SECROP_forecast_20240923_131734_best.pth',
    'LOWER5MINROP': 'lstm_LOWER5MINROP_forecast_20240923_133256_best.pth',
    'LOWERREGROP': 'lstm_LOWERREGROP_forecast_20240923_125657_best.pth'
```

## Preprocessing the LSTM price predictions.

Since the MPC takes a long time to evaluate and run, I preprocessed the price predictions into a separate pickled dictionary, where the keys are the datetime timestamps for each interval, and the values are `Prices` objects going out 288 intervals (24 hours). 

This was pre-processed using the following script, saving a large dictionary to a pickle file. The `start_date` needs to be 128 periods into the past for the lookback period, and the `end_date` needs to be 24 hours past the final interval for the optimal horizon of the final decision.

```python
from calliope.prices import construct_prices_from_merit_order_csv
prices = construct_prices_from_merit_order_csv(
    path='E:/Code/Calliope/notebooks/merit_order_prices.csv', 
    region_id= 'NSW1', 
    start_date = '2019-06-30T13:20:00', 
    end_date='2019/09/02', 
    create_agc=True
)
price_df = prices.to_dataframe()

from calliope.defaults import BEST_MODELS
from calliope.forecast import lstm
lstm.predict_prices_over_horizon(    
    BEST_MODELS, 
    prices,
    lookahead=288,
    lookback=128,
    region_id='NSW1',
    periodic_save=True,
    fn_dict_save='lstm_price_predictions.pkl'
)
```

## Running the MPC
The MPC baseline can finally be run using the following code. Note if you don't have Gurobi, you can still run the Google OR-Tools version of the code.

```python
import sqlite3
from calliope.data import DataManager
from calliope.market import MeritOrderNEM
import pandas as pd
import datetime

con = sqlite3.connect('E:/Code/Calliope/notebooks/historical_mms.db')
start_date = '2019/07/01 00:00:00'
end_date = '2019/09/01 00:00:00'
region = 'NSW1'


print('Loading data')
data_manager = DataManager(con)
data_manager.load_data(start_date, end_date, region, nem_solve_data=True)
nem = MeritOrderNEM(data_manager)
nem.build()

nem.load_bids_from_pickle(
    [f'E:/Code/Calliope/notebooks/bid_dict_{i}_nsw.pkl' for i in ['jul','aug']]
)

from calliope.optimisation import config, mip_joint_simple
from calliope import mpc
from calliope.optimisation import mip_joint_gurobi

from calliope.prices import construct_prices_from_merit_order_csv
prices = construct_prices_from_merit_order_csv(
    path='E:/Code/Calliope/notebooks/merit_order_prices.csv', 
    region_id= 'NSW1', 
    start_date = start_date,
    end_date = end_date,
    create_agc=True
)
bess_config = config.load_config('E:/Code\Calliope/scripts/config.yml')

model = mip_joint_gurobi.MIPJointBESSGurobi(bess_config)

# This can be replaced with mip_joint_simple if you don't have gurobi, it'll just take way longer
model_no_gurobi = mip_joint_simple.MixedIntegerESSModelJoint(bess_config, solver_backend='SCIP')


import pickle
with open('lstm_price_predictions.pkl', 'rb') as f:
    lstm_price_predictions = pickle.load(f)

results=mpc.run_bess_mpc_simulation(model, prices, nem, horizon=288, price_forecasts=lstm_price_predictions)
revenues = mpc.calculate_simulation_cumulative_revenue(results)
```


# Tasks
The `invoke` package is used to run any CLI util tasks in `tasks.py`.

## Black formatting
To format the codebase according to black:
```bash
invoke black
```

## Kernel
Install an IPython kernel to be used
```bash
invoke kernel
```

# TODO
- [x] Train over joint merit order model env.
- [x] Implement a custom market model in Gurobi to replace nempy. (DONE but unused)
- [x] Implement rolling horizon MIP model runner to compare DRL against.
- [x] Reimplement BESS optimisation model for MPC in Gurobi to speed things up.
- [x] Clean this whole repository up because it's a mess lmao.
