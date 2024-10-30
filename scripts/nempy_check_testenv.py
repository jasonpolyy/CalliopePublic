from stable_baselines3.common.env_checker import check_env

from calliope.envs.nempy_hist_test_env import NempyHistoricalEnv

from nempy.historical_inputs import mms_db, xml_cache
import sqlite3

# from mip import gurobi

con = sqlite3.connect('notebooks/historical_mms.db')
mms_db_manager = mms_db.DBManager(connection=con)
xml_cache_manager = xml_cache.XMLCacheManager('notebooks/nemde_cache')

dispatch_intervals = ['2024/01/01 12:00:00',
                      '2024/01/01 12:05:00',
                      '2024/01/01 12:10:00',
                      '2024/01/01 12:15:00',
                      '2024/01/01 12:20:00',
                      '2024/01/01 12:25:00',
                      '2024/01/01 12:30:00']

env = NempyHistoricalEnv(
    mms_db_manager=mms_db_manager, 
    xml_cache_manager=xml_cache_manager, 
    dispatch_intervals=dispatch_intervals, 
    solver_name='GUROBI'
)

check_env(env)

