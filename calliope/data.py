"""
Data utilities module for getting NEM data from nempy SQLite db.
"""
import sqlite3
import pandas as pd
from typing import List, Optional, Dict, Any
import datetime
from calliope.defaults import FCAS_SERVICES

DATE_FORMAT = '%Y/%m/%d %H:%M:%S'

def vali_date(date_str, date_format = DATE_FORMAT):
     return bool(datetime.datetime.strptime(date_str, date_format))

class EmptyDataFrameException(Exception):
    """Exception raised when dataframe is empty in DataManager."""

def to_period_id(hour, minute, intervals_per_hour=12):
    """Function to get the period id given the hour and minute"""
    return intervals_per_hour * hour + (minute/int(60/intervals_per_hour))

class DataManager():
    """
    Manage all data for the market simulation
    """
    def __init__(self, con: sqlite3.Connection):
        self.con_set = False
        self.set_connection(con)
        self.unit_mapping: pd.DataFrame = pd.DataFrame()
        self.price_bids: pd.DataFrame = pd.DataFrame()
        self.volume_bids: pd.DataFrame = pd.DataFrame()
        self.regional_demand: pd.DataFrame = pd.DataFrame()
        self.fcas_requirements: pd.DataFrame = pd.DataFrame()
        self.dispatch_intervals: list = []
        
    def load_data(self, start_date, end_date, region, nem_solve_data = False, verbose=False):
        """
        Load all data into the instance variables.

        If nem_solve_data=True, then only loads data needed for MeritOrderNEM.
        """
        if not self.con_set:
            raise ValueError('SQlite connection has not been set yet.')
        if not any([vali_date(start_date), vali_date(end_date)]):
            raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')

        # load data in
        self.unit_mapping = get_unit_region_mapping(self.con)

        # only care about region and generators (TOTALDEMAND nets off load so will never be correct merit order)
        mask = (self.unit_mapping.REGIONID == region) & (self.unit_mapping.DISPATCHTYPE == 'GENERATOR')
        self.unit_mapping = self.unit_mapping.loc[mask]
        if verbose: print('Loaded unit_mapping.')

        if not nem_solve_data:
            self.price_bids = get_price_bids(self.con, start_date, end_date, self.unit_mapping)
            if verbose: print('Loaded price_bids.')
            self.volume_bids = get_volume_bids(self.con, start_date, end_date, self.unit_mapping)
            if verbose: print('Loaded volume_bids.')
        else:
            self.price_bids = None
            self.volume_bids = None

        # flip load bids to negative to represent adding to regional demand
        # mask = (self.volume_bids.DISPATCHTYPE == 'LOAD') & (self.volume_bids.BIDTYPE == 'ENERGY') 
        # cols = [f'BANDAVAIL{i}' for i in range(1,11)]
        # self.volume_bids.loc[mask, cols] = self.volume_bids.loc[mask, cols] * -1 

        self.regional_demand = get_region_demand(self.con, start_date, end_date, region)
        if verbose: print('Loaded regional_demand.')

        self.regional_demand_fc = get_region_demand_forecast(self.con, start_date, end_date, region)
        if verbose: print('Loaded regional_demand_fc.')

        self.fcas_requirements = get_fcas_requirements_region_assumed(self.con, start_date, end_date, region)
        if verbose: print('Loaded fcas_requirements.')

        # the forecast offset for error is just from the demand forecast in this case
        # this should be a percentage change
        forecast_offset_perc = self.regional_demand['DEMANDFORECAST'] / self.regional_demand['TOTALDEMAND']

        # modify fcas_requirements with demand forecast operational demand difference
        self.fcas_requirements_fc = get_fcas_requirements_region_assumed_forecast_adjusted(
            self.con, start_date, end_date, region, 
            forecast_offset_perc=forecast_offset_perc
        )
        if verbose: print('Loaded fcas_requirements_fc.')

        interval_datetimes = get_interval_datetime_from_volume_bids(self.con, start_date, end_date)
        self.dispatch_intervals = interval_datetimes.INTERVAL_DATETIME.unique().tolist()
        self.di_periodid = [to_period_id(i.hour, i.minute)  for i in self.dispatch_intervals]

        self.check_data_loaded()

    def set_connection(
            self, 
            con: sqlite3.Connection
        ):
        """
        Add an sqlite3.Connection object to the datamanager.
        This method exists to handle the case datamanager is reloaded from pickle.
        This is because a connection object cannot be pickled.

        Parameters
        ----------
        con: sqlite3.Connection
            the connection object
        """
        self.con = con
        self.con_set = True

    def check_data_loaded(self):
        """
        Check that all data has been loaded by checking if dataframes are empty.
        Expose public method so data manager can be publically verified as loaded.
        """
        for k, v in self.__dict__.items():
            
            if isinstance(v, pd.DataFrame):
                if v.empty:
                    raise EmptyDataFrameException(f'self.{k} is an empty DataFrame.')
            
    def __getstate__(self) -> Dict[str, Any]:
        """
        Get the state of the object. This is what `pickle` sees when it pickles an object.
        """
        return {k:v for (k, v) in self.__dict__.items() if self._picklable(k)}
    
    def _picklable(self, item):
        """
        Terrible boolean check to see if item can be pickled.
        """
        if item == 'con':
            return False
        return True
            

def get_unit_region_mapping(
    con: sqlite3.Connection,
):
    """
    Get unit to region mapping to more easily filter out unused units.
    """
    query = f"""select distinct DUID, REGIONID, DISPATCHTYPE from DUDETAILSUMMARY"""
    data = pd.read_sql(query, con=con)
    return data

def get_price_bids(
    con: sqlite3.Connection,
    start_date: str,
    end_date: str,
    duids: Optional[List[str]] = None
):
    """
    Get daily price bids for all units.
    """
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')
    
    if duids is None:
        duids = get_unit_region_mapping(con)

        # only getting generators here
        duids = duids.loc[duids.DISPATCHTYPE=='GENERATOR']
    duids = duids.DUID.tolist()

    query = """
    select 
    SETTLEMENTDATE, DUID, BIDTYPE, 
    PRICEBAND1, PRICEBAND2, PRICEBAND3, PRICEBAND4, PRICEBAND5, 
    PRICEBAND6, PRICEBAND7, PRICEBAND8, PRICEBAND9, PRICEBAND10
    from BIDDAYOFFER_D
    where duid in ({})
    and settlementdate between ? and ?
    """.format(', '.join(['?' for _ in duids]))

    param_list = duids+[start_date, end_date]
    df = pd.read_sql(query, con=con, params=param_list)

    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], format=DATE_FORMAT)

    return df

def get_interval_datetime_from_volume_bids(   
    con: sqlite3.Connection,
    start_date: str,
    end_date: str
):
    """
    Get 5min volume bids for all units.
    """
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')
    query = """
    select 
    distinct INTERVAL_DATETIME
    from BIDPEROFFER_D
    where INTERVAL_DATETIME between ? and ?
    """
    param_list = [start_date, end_date]
    df = pd.read_sql(query, con=con, params=param_list)
    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format=DATE_FORMAT)
    
    return df

def get_volume_bids(
    con: sqlite3.Connection,
    start_date: str,
    end_date: str,
    duids: Optional[List[str]] = None
):
    """
    Get 5min volume bids for all units.
    """
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')

    if duids is None:
        duids = get_unit_region_mapping(con)

        # only getting generators here
        duids = duids.loc[duids.DISPATCHTYPE=='GENERATOR']
    gen_mapping = duids[['DUID', 'DISPATCHTYPE']]
    duids = duids.DUID.tolist()

    query = """
    select 
    INTERVAL_DATETIME, DUID, BIDTYPE,
    BANDAVAIL1, BANDAVAIL2, BANDAVAIL3, BANDAVAIL4, BANDAVAIL5,
    BANDAVAIL6, BANDAVAIL7, BANDAVAIL8, BANDAVAIL9, BANDAVAIL10,
    MAXAVAIL, ENABLEMENTMIN, ENABLEMENTMAX, LOWBREAKPOINT, HIGHBREAKPOINT
    from BIDPEROFFER_D
    where duid in ({})
    and INTERVAL_DATETIME between ? and ?
    """.format(', '.join(['?' for _ in duids]))

    param_list = duids+[start_date, end_date]
    df = pd.read_sql(query, con=con, params=param_list)

    df = pd.merge(df, gen_mapping, how='left', on='DUID')
    
    # make bids negative for loads as per merit order model
    # mask = (df.DISPATCHTYPE == 'LOAD') & (df.BIDTYPE == 'ENERGY') 
    # cols = [f'BANDAVAIL{i}' for i in range(1,11)]
    # df.loc[mask, cols] = df.loc[mask, cols] * -1 

    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format=DATE_FORMAT)
    
    return df

def get_region_demand(
    con: sqlite3.Connection,
    start_date: str,
    end_date: str,
    region: str
):
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')

    query = """
    select 
    SETTLEMENTDATE,	REGIONID,	TOTALDEMAND,	DEMANDFORECAST,	INITIALSUPPLY,
    TOTALDEMAND+DEMANDFORECAST as TOTALDEMAND_FC
    from DISPATCHREGIONSUM
    where regionid = ?
    and settlementdate between ? and ?

    """
    df = pd.read_sql(query, con=con, params=[region, start_date, end_date])

    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], format=DATE_FORMAT)
    return df

def get_region_demand_forecast(
    con: sqlite3.Connection,
    start_date: str,
    end_date: str,
    region: str
):
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')

    end_date = datetime.datetime.strftime(
        datetime.datetime.strptime(end_date, DATE_FORMAT) + datetime.timedelta(0,300), 
        DATE_FORMAT
    )

    query = """
    select 
    SETTLEMENTDATE,	REGIONID,	TOTALDEMAND,	DEMANDFORECAST,	INITIALSUPPLY,
    TOTALDEMAND+DEMANDFORECAST as TOTALDEMAND_FC
    from DISPATCHREGIONSUM
    where regionid = ?
    and settlementdate between ? and ?

    """
    df = pd.read_sql(query, con=con, params=[region, start_date, end_date])

    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], format=DATE_FORMAT)

    # remove a 5min period
    df['SETTLEMENTDATE'] -=  datetime.timedelta(0, 60*5)
    df = df.iloc[1:]

    df = df.reset_index(drop=True)
    
    return df

def get_fcas_requirements_region_assumed(
    con: sqlite3.Connection,
    start_date: str,
    end_date: str,
    region: str,
    duids: Optional[List[str]] = None
):
    """
    Get assumed FCAS requirements per region. This only holds under the assumption
    that regional dispatched amount for FCAS = requirement.

    Actual FCAS requirements are procured based on generic LHS and RHS constraints
    as exampled in nempy. This approximation is such that the data can be easily
    obtained for the purpose of demonstrative market simulations.
    """
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')
    
    if duids is None:
        duids = get_unit_region_mapping(con)
        duids_region = duids.loc[duids.REGIONID == region].DUID.tolist()

    query = """
    select 
    SETTLEMENTDATE,
    sum(RAISEREG) as RAISEREG, 
    sum(LOWERREG) as LOWERREG, 
    sum(RAISE6SEC) as RAISE6SEC, 
    sum(RAISE60SEC) as RAISE60SEC, 
    sum(RAISE5MIN) as RAISE5MIN,
    sum(LOWER6SEC) as LOWER6SEC, 
    sum(LOWER60SEC) as LOWER60SEC,
    sum(LOWER5MIN) as LOWER5MIN
    from DISPATCHLOAD
    where duid in ({})
    and settlementdate between ? and ?
    group by settlementdate
    """.format(', '.join(['?' for _ in duids_region]))

    param_list = duids_region+[start_date, end_date]
    df = pd.read_sql(query, con=con, params=param_list)
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], format=DATE_FORMAT)

    return df

def get_fcas_requirements_region_assumed_forecast_adjusted(
    con: sqlite3.Connection,
    start_date: str,
    end_date: str,
    region: str,
    duids: Optional[List[str]] = None,
    forecast_offset_perc: pd.DataFrame = None
):
    """
    Get assumed FCAS requirements per region. This only holds under the assumption
    that regional dispatched amount for FCAS = requirement.

    This function differs by shifting the date index by one so that the settlementdate is for
    the next interval, but some forecast error is added based on the demand forecast difference
    observed in the operational demand dataset.

    Actual FCAS requirements are procured based on generic LHS and RHS constraints
    as exampled in nempy. This approximation is such that the data can be easily
    obtained for the purpose of demonstrative market simulations.
    """
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')
    
    if duids is None:
        duids = get_unit_region_mapping(con)
        duids_region = duids.loc[duids.REGIONID == region].DUID.tolist()

    end_date = datetime.datetime.strftime(
        datetime.datetime.strptime(end_date, DATE_FORMAT) + datetime.timedelta(0,300), 
        DATE_FORMAT
    )

    query = """
    select 
    SETTLEMENTDATE,
    sum(RAISEREG) as RAISEREG, 
    sum(LOWERREG) as LOWERREG, 
    sum(RAISE6SEC) as RAISE6SEC, 
    sum(RAISE60SEC) as RAISE60SEC, 
    sum(RAISE5MIN) as RAISE5MIN,
    sum(LOWER6SEC) as LOWER6SEC, 
    sum(LOWER60SEC) as LOWER60SEC,
    sum(LOWER5MIN) as LOWER5MIN
    from DISPATCHLOAD
    where duid in ({})
    and settlementdate between ? and ?
    group by settlementdate
    """.format(', '.join(['?' for _ in duids_region]))

    param_list = duids_region+[start_date, end_date]
    df = pd.read_sql(query, con=con, params=param_list)
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], format=DATE_FORMAT)

    # remove a 5min period
    df['SETTLEMENTDATE'] -=  datetime.timedelta(0, 60*5)
    df = df.iloc[1:]

    df = df.reset_index(drop=True)

    if forecast_offset_perc is not None:
        for m in FCAS_SERVICES:
            df[f'{m}_FC'] = df[f'{m}'] + df[f'{m}']*forecast_offset_perc

    return df


def get_volume_bids_long(
    con: sqlite3.Connection,
    start_date: str,
    end_date: str,
    duids: Optional[List[str]] = None
):
    """
    Get 5min volume bids for all units.
    """
    if not any([vali_date(start_date), vali_date(end_date)]):
        raise ValueError(f'Dates not in correct format. Must be {DATE_FORMAT}')

    if duids is None:
        duids = get_unit_region_mapping(con)
    gen_mapping = duids[['DUID', 'DISPATCHTYPE']]
    duids = duids.DUID.tolist()

    query = """
    with cte as (
        select 
        INTERVAL_DATETIME, DUID, BIDTYPE,
        BANDAVAIL1, BANDAVAIL2, BANDAVAIL3, BANDAVAIL4, BANDAVAIL5,
        BANDAVAIL6, BANDAVAIL7, BANDAVAIL8, BANDAVAIL9, BANDAVAIL10
        from BIDPEROFFER_D
        where duid in ({})
        and INTERVAL_DATETIME between ? and ?
    )

    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 1 AS BAND, BANDAVAIL1 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 2 AS BAND, BANDAVAIL2 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 3 AS BAND, BANDAVAIL3 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 4 AS BAND, BANDAVAIL4 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 5 AS BAND, BANDAVAIL5 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 6 AS BAND, BANDAVAIL6 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 7 AS BAND, BANDAVAIL7 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 8 AS BAND, BANDAVAIL8 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 9 AS BAND, BANDAVAIL9 AS Q_BID FROM CTE UNION ALL
    SELECT INTERVAL_DATETIME, DUID, BIDTYPE, 10 AS BAND, BANDAVAIL10 AS Q_BID FROM CTE;
    """.format(', '.join(['?' for _ in duids]))

    param_list = duids+[start_date, end_date]
    df = pd.read_sql(query, con=con, params=param_list)

    df = pd.merge(df, gen_mapping, how='left', on='DUID')
    
    # make bids negative for loads as per merit order model
    mask = (df.DISPATCHTYPE == 'LOAD') & (df.BIDTYPE == 'ENERGY') 
    #cols = [f'BANDAVAIL{i}' for i in range(1,11)]
    df.loc[mask, 'Q_BID'] = df.loc[mask, 'Q_BID'] * -1 

    df['INTERVAL_DATETIME'] = pd.to_datetime(df['INTERVAL_DATETIME'], format=DATE_FORMAT)
    
    return df

