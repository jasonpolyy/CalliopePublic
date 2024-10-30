from stable_baselines3.common.env_checker import check_env

# from calliope.nempy_hist_test_env import NempyHistoricalEnv
# from calliope.nempy_hist_simple_env import NempyHistoricalSimpleEnv
from calliope.envs.nempy_hist_selfschedule import NempyHistoricalSelfScheduleEnv

from nempy.historical_inputs import mms_db, xml_cache
import sqlite3

from calliope.optimisation.config import load_config

import datetime
# from mip import gurobi

con = sqlite3.connect('notebooks/historical_mms.db')
mms_db_manager = mms_db.DBManager(connection=con)
xml_cache_manager = xml_cache.XMLCacheManager('notebooks/nemde_cache')

start_time = datetime.datetime(year=2024, month=1, day=1, hour=12, minute=0)
end_time = datetime.datetime(year=2024, month=1, day=1, hour=12, minute=30)

bess_config = load_config('E:/Code/Calliope/scripts/config.yml')

env = NempyHistoricalSelfScheduleEnv(
    mms_db_manager=mms_db_manager,
    xml_cache_manager=xml_cache_manager,
    start_time=start_time,
    end_time=end_time,
    solver_name='GUROBI',
    bess_config=bess_config
)

check_env(env)

