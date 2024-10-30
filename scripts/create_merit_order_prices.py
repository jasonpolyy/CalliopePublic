import pickle
from calliope.data import DataManager
import pandas as pd
import sqlite3
con = sqlite3.connect("C:/Users/YOUR_PATH/historical_mms.db")
data_manager = DataManager(con)
from calliope.market import MeritOrderNEM

# Create the merit order raw prices to train over
start_date = '2019/01/01 00:00:00'
end_date = '2019/11/01 00:00:00'
region = 'NSW1'

if __name__ == '__main__':

    data_manager.load_data(start_date, end_date, region)
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug','sep','oct']
    bid_dict_files=[f'E:/Code/Calliope/notebooks/bid_dict_{i}_nsw.pkl' for i in months]
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