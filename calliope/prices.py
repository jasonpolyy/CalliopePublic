"""
Data classes for spot prices to be used in optimisation models.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional
import copy

@dataclass
class Prices:
    idx: np.array
    SETTLEMENTDATE: np.array
    REGIONID: np.array
    ROP: np.array
    RAISE6SECROP: np.array
    RAISE60SECROP: np.array
    RAISE5MINROP: np.array
    RAISEREGROP: np.array
    LOWER6SECROP: np.array
    LOWER60SECROP: np.array
    LOWER5MINROP: np.array
    LOWERREGROP: np.array
    RAISE1SECROP: Optional[np.array] = None
    LOWER1SECROP: Optional[np.array] = None
    AGC: Optional[np.array] = None

    def __post_init__(self):
        """
        Post processing for init method
        """
        self.intervals_per_hour = get_intervals_per_hour(self.SETTLEMENTDATE)  
        
    def to_matrix(self):
        """Convert to a matrix for easy extraction"""
        m = np.matrix(
            [
                self.ROP.astype('float32'),
                self.RAISE6SECROP.astype('float32'),
                self.RAISE60SECROP.astype('float32'),
                self.RAISE5MINROP.astype('float32'),
                self.RAISEREGROP.astype('float32'),
                self.LOWER6SECROP.astype('float32'),
                self.LOWER60SECROP.astype('float32'),
                self.LOWER5MINROP.astype('float32'),
                self.LOWERREGROP.astype('float32'),
            ]
        ).transpose().astype('float32')
        return m

    def to_dataframe(self):
        """
        Transform into a dataframe object again.
        """
        df = pd.DataFrame(
            {
                "SETTLEMENTDATE": self.SETTLEMENTDATE,
                "REGIONID": self.REGIONID,
                "ROP": self.ROP,
                "RAISE6SECROP": self.RAISE6SECROP,
                "RAISE60SECROP": self.RAISE60SECROP,
                "RAISE5MINROP": self.RAISE5MINROP,
                "RAISEREGROP": self.RAISEREGROP,
                "LOWER6SECROP": self.LOWER6SECROP,
                "LOWER60SECROP": self.LOWER60SECROP,
                "LOWER5MINROP": self.LOWER5MINROP,
                "LOWERREGROP": self.LOWERREGROP,
            },
            index=self.idx,
        )

        if self.RAISE1SECROP is not None:
            df["RAISE1SECROP"] = self.RAISE1SECROP
        if self.LOWER1SECROP is not None:
            df["LOWER1SECROP"] = self.LOWER1SECROP

        return df

def construct_prices_from_dict(
        price_dict: dict,
        region_id: str
):
    """
    Create a Prices object from a dictionary where all keys are the price data for arrays
    """
    # get len to get idx
    price_dict['idx'] = np.array(range(len(price_dict['ROP'])))
    price_dict['REGIONID'] = region_id

    price_dict['AGC'] = np.random.uniform(-1,1, size=len(price_dict['ROP']))
    price_dict['RAISE1SECROP'] = None
    price_dict['LOWER1SECROP'] = None

    return Prices(**price_dict)

def get_prices_over_horizon(prices: Prices, horizon_start, horizon_end) -> Prices:
    """
    Return a copy of the input prices, where all values have been filtered
    to the horizon by the idx.
    """
    prices_new = copy.deepcopy(prices)
    prices_new.idx = prices_new.idx[horizon_start:horizon_end]
    prices_new.SETTLEMENTDATE = prices_new.SETTLEMENTDATE[horizon_start:horizon_end]
    prices_new.REGIONID = prices_new.REGIONID[horizon_start:horizon_end]
    prices_new.ROP = prices_new.ROP[horizon_start:horizon_end]
    prices_new.RAISE6SECROP = prices_new.RAISE6SECROP[horizon_start:horizon_end]
    prices_new.RAISE60SECROP = prices_new.RAISE60SECROP[horizon_start:horizon_end]
    prices_new.RAISE5MINROP = prices_new.RAISE5MINROP[horizon_start:horizon_end]
    prices_new.RAISEREGROP = prices_new.RAISEREGROP[horizon_start:horizon_end]
    prices_new.LOWER6SECROP = prices_new.LOWER6SECROP[horizon_start:horizon_end]
    prices_new.LOWER60SECROP = prices_new.LOWER60SECROP[horizon_start:horizon_end]
    prices_new.LOWER5MINROP = prices_new.LOWER5MINROP[horizon_start:horizon_end]
    prices_new.LOWERREGROP = prices_new.LOWERREGROP[horizon_start:horizon_end]
    prices_new.AGC = prices_new.AGC[horizon_start:horizon_end]
    return prices_new
  

def get_intervals_per_hour(SETTLEMENTDATE: np.array):
    """
    Get the number of time intervals in an hour. 

    This is calculated as the difference in minutes between the first two settlementdate objects,
    and then converted to a numeric value via the reciprocal.

    Parameters
    ----------
    SETTLEMENTDATE : np.array
        SETTLEMENTDATE array to get the number of periods.
    """
    return 1/((SETTLEMENTDATE[1] - SETTLEMENTDATE[0]).astype('timedelta64[m]').astype('int')/60)



def construct_prices_from_merit_order_csv(
    path: str,
    region_id: str,
    start_date: str,
    end_date: str, 
    create_agc: bool = False
) -> Prices:
    """
    Load CSV prices generated from the Merit Order NEM model into a Prices object.

    Parameters
    ----------
    path : str
        path to the csv
    region_id : str
        the region of the prices
    create_agc : bool, optional
        whether to create agc signals to follow, by default False

    Returns
    -------
    Prices
        _description_
    """
    data = pd.read_csv(path)
    agc = None
    r1s, l1s = None, None

    # Perform date conversion
    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["timestamp"], format="%Y-%m-%d %H:%M:%S"
    )

    data['REGIONID'] = region_id
 
    data = data.loc[(data.SETTLEMENTDATE >= start_date) & (data.SETTLEMENTDATE <= end_date)]

    if create_agc:
        agc = np.random.uniform(-1,1, size=data.index.shape[0])

    data = data.reset_index(drop=True)

    prices = Prices(
        idx=data.index.to_numpy(),
        SETTLEMENTDATE=data["SETTLEMENTDATE"].to_numpy(),
        REGIONID=data["REGIONID"].to_numpy(),
        ROP=data["ENERGY"].to_numpy(),
        RAISE6SECROP=data["RAISE6SEC"].to_numpy(),
        RAISE60SECROP=data["RAISE60SEC"].to_numpy(),
        RAISE5MINROP=data["RAISE5MIN"].to_numpy(),
        RAISEREGROP=data["RAISEREG"].to_numpy(),
        LOWER6SECROP=data["LOWER6SEC"].to_numpy(),
        LOWER60SECROP=data["LOWER60SEC"].to_numpy(),
        LOWER5MINROP=data["LOWER5MIN"].to_numpy(),
        LOWERREGROP=data["LOWERREG"].to_numpy(),
        RAISE1SECROP=r1s,
        LOWER1SECROP=l1s,
        AGC=agc
    )

    return prices



def construct_prices_from_mms(
    con: sqlite3.Connection,
    region_id: str,
    start_date: str,
    end_date: str,
    FFR: bool = False,
    create_agc: bool = False,
) -> Prices:
    """
    Construct a Prices object from AEMO MMS SQLite database.

    Parameters
    ----------
    con : sqlite3.Connection
        connection to SQLite database
    """
    query = f"""
    select 
        SETTLEMENTDATE,
        REGIONID,
        ROP,
        RAISE6SECROP,
        RAISE1SECROP,
        RAISE60SECROP,
        RAISE5MINROP,
        RAISEREGROP,
        LOWER6SECROP,
        LOWER1SECROP,
        LOWER60SECROP,
        LOWER5MINROP,
        LOWERREGROP
    from
        DISPATCHPRICE
    where regionid = ?
    and settlementdate between ? and ?
    """
    data = pd.read_sql(query, con=con, params=[region_id, start_date, end_date])

    r1s, l1s = None, None
    agc = None
    if FFR:
        r1s = data["RAISE1SECROP"].to_numpy()
        l1s = data["LOWER1SECROP"].to_numpy()

    # Perform date conversion
    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S"
    )

    if create_agc:
        agc = np.random.uniform(-1,1, size=data.index.shape[0])

    prices = Prices(
        idx=data.index.to_numpy(),
        SETTLEMENTDATE=data["SETTLEMENTDATE"].to_numpy(),
        REGIONID=data["REGIONID"].to_numpy(),
        ROP=data["ROP"].to_numpy(),
        RAISE6SECROP=data["RAISE6SECROP"].to_numpy(),
        RAISE60SECROP=data["RAISE60SECROP"].to_numpy(),
        RAISE5MINROP=data["RAISE5MINROP"].to_numpy(),
        RAISEREGROP=data["RAISEREGROP"].to_numpy(),
        LOWER6SECROP=data["LOWER6SECROP"].to_numpy(),
        LOWER60SECROP=data["LOWER60SECROP"].to_numpy(),
        LOWER5MINROP=data["LOWER5MINROP"].to_numpy(),
        LOWERREGROP=data["LOWERREGROP"].to_numpy(),
        RAISE1SECROP=r1s,
        LOWER1SECROP=l1s,
        AGC=agc
    )

    return prices
