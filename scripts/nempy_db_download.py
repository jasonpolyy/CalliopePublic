import sqlite3
from datetime import datetime, timedelta
import random
import pandas as pd
from nempy import markets
from nempy.historical_inputs import  mms_db, xml_cache
from nempy.historical_inputs.mms_db import InputsByDay, InputsByIntervalDateTime

import argparse
import datetime

def populate_bid_tables(
        db_manager, 
        years=[2019], 
        months=range(1,13), 
        load_bidday=False,
        load_bidper=False
):
    """
    Helper function for populating bid tables as this functionality was taken out in later
    nempy releases due to Oct21 5ms data change to AEMO MMS.
    """

    if load_bidday:
        db_manager.BIDDAYOFFER_D = InputsByDay(
            table_name='BIDDAYOFFER_D', table_columns=['SETTLEMENTDATE', 'DUID', 'BIDTYPE', 'PRICEBAND1', 'PRICEBAND2',
                                                        'PRICEBAND3', 'PRICEBAND4', 'PRICEBAND5', 'PRICEBAND6',
                                                        'PRICEBAND7', 'PRICEBAND8', 'PRICEBAND9', 'PRICEBAND10', 'T1',
                                                        'T2', 'T3', 'T4', 'MINIMUMLOAD'],
            table_primary_keys=['SETTLEMENTDATE', 'DUID', 'BIDTYPE'], con=con)
        
    if load_bidper:
        db_manager.BIDPEROFFER_D = InputsByIntervalDateTime(
                    table_name='BIDPEROFFER_D', table_columns=['INTERVAL_DATETIME', 'DUID', 'BIDTYPE', 'BANDAVAIL1',
                                                            'BANDAVAIL2', 'BANDAVAIL3', 'BANDAVAIL4', 'BANDAVAIL5',
                                                            'BANDAVAIL6', 'BANDAVAIL7', 'BANDAVAIL8', 'BANDAVAIL9',
                                                            'BANDAVAIL10', 'MAXAVAIL', 'ENABLEMENTMIN', 'ENABLEMENTMAX',
                                                            'LOWBREAKPOINT', 'HIGHBREAKPOINT'],
                    table_primary_keys=['INTERVAL_DATETIME', 'DUID', 'BIDTYPE'], con=con)
    
    for year in years:
        for month in months:
            print(f'Loading data for {month}')

            try:
                if load_bidday: db_manager.BIDDAYOFFER_D.add_data(year=year, month=month)
                if load_bidper: db_manager.BIDPEROFFER_D.add_data(year=year, month=month)
            except:
                print('No data available for this period')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str,help='Path to save the data.')
    parser.add_argument('--download_inputs', default=False, type=bool,help='Whether to download inputs or not.')

    args = parser.parse_args()
    db_path = args.path
    download_inputs = args.download_inputs

    con = sqlite3.connect(f'{db_path}/historical_mms.db')
    mms_db_manager = mms_db.DBManager(connection=con)

    # The second time this example is run on a machine this flag can
    # be set to false to save downloading the data again.

    if download_inputs:
        
        # This requires approximately ? gb of storage!
        mms_db_manager.populate(start_year=2019, start_month=1,
                                end_year=2020, end_month=1)


        # Check whether biddayoffer_d and bidperoffer_d have been populated
        load_bidday = False
        load_bidper = False
        query = """select * from BIDDAYOFFER_D limit 1"""
        try:
            df = pd.read_sql(query, con=con)
        except Exception as e:
            load_bidday = True
            print(e)
        query = """select * from BIDPEROFFER_D limit 1"""
        try:
            df = pd.read_sql(query, con=con)
        except Exception as e:
            load_bidper = True
            print(e)

        if load_bidday or load_bidper:
            populate_bid_tables(
                mms_db_manager, 
                years=[2019], 
                months=range(1,13), 
                load_bidday=load_bidday,
                load_bidper=load_bidper
            )

        




