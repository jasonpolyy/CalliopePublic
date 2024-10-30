import pandas as pd
import numpy as np
import datetime
from typing import List

# default price bids by taking percentiles of rop, PB1 and PB10 are MFC and MPC respectively
PRICE_BIDS_DEFAULT = np.array(
    [
        -9.95783500e+02, 
        -4.71379030e+01,  
        8.95000000e+00,
        2.28149960e+01,
        5.38129200e+01,
        7.34002820e+01,
        9.00000000e+01,
        1.11627174e+02,
        1.43719486e+02,
        1.45222563e+04
    ]
)

def get_next_dispatch_interval(time: datetime.datetime) -> str:
    new_time = time + datetime.timedelta(minutes=5)
    new_time_fmt = new_time.isoformat().replace('T', ' ').replace('-', '/') 
    return new_time_fmt

def get_dispatch_intervals(start_time: datetime.datetime, end_time: datetime.datetime) -> List[str]:
    """
    Get dispatch intervals between two datetimes as a list for intervals we want to recreate historical dispatch for.

    Parameters
    ----------
    start_time : datetime.datetime
        start time for the nempy horizon
    end_time : datetime.datetime
        end time for the nempy horizon

    Returns
    -------
    List[str]
        list of strings in the format '%Y/%m/%d %H:%M:%S'
    """
    # start_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
    # end_time = datetime.datetime(year=2024, month=1, day=31, hour=0, minute=0)
    difference = end_time - start_time
    difference_in_5_min_intervals = difference.days*24*12 + int(difference.seconds / 300) +1
    times = [start_time + datetime.timedelta(minutes=5 * i) for i in range(difference_in_5_min_intervals)]
    times_formatted = [t.isoformat().replace('T', ' ').replace('-', '/') for t in times]
    return times_formatted

def format_unit_info(unit, dispatch_type, region, loss_factor):
    """
    Format the unit information to be added to the SpotMarket object.
    Doesn't include connection point as unneeded.
    >>> print(unit_info)
      unit dispatch_type  region  loss_factor
    0    A     generator    NSW1       0.8100
    1    B          load     SA1       0.8415
    """
    unit_info = pd.DataFrame({
        'unit': [unit],
        'dispatch_type': [dispatch_type],
        'region': [region],
        'loss_factor': [loss_factor],
    })

    return unit_info


def format_volume_bids(unit, market, volumes):
    """
    Format the volume bids for
    """

def extract_region_price(prices, region):
    mask = (prices.region == region)
    df = prices.loc[mask]
    if df.empty:
        return 0.0
    return df.price.reset_index(drop=True).values[0]

def extract_region_demand(demand, region):
    mask = (demand.region == region)
    df = demand.loc[mask]
    if df.empty:
        return 0.0
    return df.demand.reset_index(drop=True).values[0]

def extract_fcas_price(prices, service):
    mask = (prices.service == service)
    df = prices.loc[mask]
    if df.empty:
        return 0.0
    return df.price.reset_index(drop=True).values[0]

def extract_unit_dispatch(dispatch, service, unit):
    mask = (dispatch.unit == unit) & (dispatch.service == service)
    df = dispatch.loc[mask]
    if df.empty:
        return 0.0
    return df.dispatch.reset_index(drop=True).values[0]