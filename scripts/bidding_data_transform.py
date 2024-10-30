import pickle
from calliope.data import DataManager
from calliope.market import DataToBidObjInterface
import sqlite3
import datetime

def tomnthstr(mnth):
    return f'0{mnth}' if mnth <10 else f'{mnth}'

if __name__ == '__main__':

    con = sqlite3.connect("C:/Users/YOUR_PATH/historical_mms.db")

    data_manager = DataManager(con)

    region = 'NSW1'
    mnths = range(1,11)
    months = ['jan', 'feb', 'mar','apr', 'may', 'jun', 'jul','aug','sep','oct']

    for mnth, mn in zip(mnths, months):
        print(f'Running month {tomnthstr(mnth)}')
        start_date = f'2019/{tomnthstr(mnth)}/01 00:00:00'
        end_date = f'2019/{tomnthstr(mnth+1)}/01 00:00:00'
        print(f'Loading data.')
        data_manager.load_data(start_date, end_date, region)
        print(f'Creating bids.')
        data_to_bid = DataToBidObjInterface(data_manager)
        bid_dict = data_to_bid.data_to_bids()
        print(f'Saving to bid files')
        with open(f'bid_dict_{mn}_nsw.pkl', 'wb') as f:
            pickle.dump(bid_dict, f, pickle.HIGHEST_PROTOCOL)
        print('done')