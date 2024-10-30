"""
Module to represent energy bid stack and FCAS markets.

Code referenced from:
https://github.com/UNSW-CEEM/energy-market-deep-learning/tree/master
"""

from typing import Dict, List
import random
from calliope.data import DataManager
import pandas as pd
import datetime
import pickle

from calliope.defaults import MARKETS, PRICE_BID_COLS, VOLUME_BID_COLS, FCAS_SERVICES

class Bid():
    """A bid that represents a price-quantity pair in the system."""
    def __init__(self, duid, price, quantity, band):
        self.duid = duid
        #self._check_dispatch_type(dispatch_type)
        #self.dispatch_type = dispatch_type
        # Add a random number here so that when bids are tied, the selection is truly random. 
        self.price = price #+ random.random()
        self.quantity = quantity
        self.band = band
    
    def copy(self):
        return Bid(self.duid, self.price, self.quantity, self.band)
    
    def to_dict(self):
        return {
            'duid':self.duid,
            'price':self.price,
            'quantity':self.quantity,
            'band':self.band,
        }
    def __str__(self):
        return f'{self.duid}, {self.band}: <{self.price},{self.quantity}>'
    

class DataToBidObjInterface():
    """
    Interface between DataManager and market classes to create
    Bid objects from DataManager bid data.
    """
    def __init__(self, data_manager: DataManager):
        self.data_manager: DataManager = data_manager
        self.data_manager.check_data_loaded()

    def data_to_bids(
            self
        ) -> Dict[str, Dict[str, List[Bid]]]:
        """
        Convert price and quantity bids from the data manager to a dictionary
        mapping each bidtype market to a list of bid objects.

        Returns
        -------
        Dict[str, List[Bid]]
            return a dictionary where keys are markets and values are lists of Bid objects
        """
        volume_bids = self.data_manager.volume_bids
        price_bids = self.data_manager.price_bids

        # Interval datetime is the end of the dispatch interval. 
        # this means midnight actually belongs to the previous day, so need to account when creating settlementdate
        volume_bids['SETTLEMENTDATE'] = pd.to_datetime(
            volume_bids.INTERVAL_DATETIME.dt.date - datetime.timedelta(seconds=1)
        )

        pq_bids = pd.merge(volume_bids, price_bids, how='left', on=['SETTLEMENTDATE', 'DUID', 'BIDTYPE'])

        # convert from wide to long data format
        pq_bids_melted = self._melt_price_quantity_bids(pq_bids)

        # create dictionary indexed by interval_datetime
        bid_dict = {d: {} for d in self.data_manager.dispatch_intervals}

        for key_datetime in bid_dict:

            pq_bid_i = pq_bids_melted.loc[pq_bids_melted.INTERVAL_DATETIME == key_datetime]

            # Obscene dictionary comprehension
            # Create a dictionary indexed by the market (BIDTYPE==m), where the values are lists of Bid objects.
            bid_dict_i = {
                m: pq_bid_i.loc[pq_bid_i.BIDTYPE == m]
                    .apply(
                        lambda x: Bid(x.DUID, x.PRICE_BID, x.QUANTITY_BID, x.BAND), 
                        axis=1
                    ).tolist() 
                    for m in MARKETS
                }

            bid_dict[key_datetime] = bid_dict_i
        
        return bid_dict

    def _melt_price_quantity_bids(
            self, 
            pq_bids
        ) -> pd.DataFrame:
        """
        Takes a dataframe of wide price quantity bids where there are 10 columns
        for all 10 bid bands, and converts it into a long format 
        where BAND is a column and price and quantity bid values are 2 columns.

        Final columns: 'INTERVAL_DATETIME', 'DUID', 'BIDTYPE', 'DISPATCHTYPE', 'BAND', 'PRICE_BID', 'QUANTITY_BID'

        Parameters
        ----------
        pq_bids : pd.DataFrame
            dataframe where bands 1-10 are separate columns for both price and volume

        Returns
        -------
        pd.DataFrame
            long format with simplified columns
        """

        # Melt the PRICEBAND columns
        melted_priceband = pq_bids.melt(
            id_vars=['INTERVAL_DATETIME', 'DUID', 'BIDTYPE', 'DISPATCHTYPE'], 
            value_vars=PRICE_BID_COLS, 
            var_name='PRICEBAND', 
            value_name='PRICE_BID'
        )

        # Melt the BANDAVAIL columns
        melted_bandavail = pq_bids.melt(
            id_vars=['INTERVAL_DATETIME', 'DUID', 'BIDTYPE', 'DISPATCHTYPE'], 
            value_vars=VOLUME_BID_COLS, 
            var_name='BANDAVAIL', 
            value_name='QUANTITY_BID'
        )

        # Get just the band number itself
        melted_priceband['BAND'] = melted_priceband['PRICEBAND'].str.extract('(\d+)').astype(int)
        melted_bandavail['BAND'] = melted_bandavail['BANDAVAIL'].str.extract('(\d+)').astype(int)

        # Merge the melted dataframes
        merged_df = pd.merge(
            melted_bandavail, 
            melted_priceband, 
            left_on=['INTERVAL_DATETIME', 'DUID', 'BIDTYPE', 'DISPATCHTYPE', 'BAND'],
            right_on=['INTERVAL_DATETIME', 'DUID', 'BIDTYPE', 'DISPATCHTYPE', 'BAND']
        )

        final_columns = ['INTERVAL_DATETIME', 'DUID', 'BIDTYPE', 'DISPATCHTYPE', 'BAND', 'PRICE_BID', 'QUANTITY_BID']
        final_df = merged_df[final_columns]

        return final_df


class BidStack():
    """Provides an api that handles bidstack calculations."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.stack = []
    
    def add_price_quantity_bid(self, bid_obj):
        """Adds a price <> quantity bid for a given participant."""
        self.stack.append( bid_obj )
    
    def economic_dispatch(self, capacity_MW, verbose=False):
        """Takes a capacity_MW and returns modified bids accepted under economic dispatch."""
        meritorder = sorted(self.stack, key=lambda k: k.price)
        
        accepted = []
        cumulative_cap_MW = 0
        satisfied = False

        # Loop through the sorted bids.
        for bid in meritorder:
            if verbose: print(f'Evaluating {str(bid)}')
            if cumulative_cap_MW + bid.quantity < capacity_MW:
                cumulative_cap_MW += bid.quantity

                # load bids are simply added to demand (or subtract from cumulative MW to move bids up merit order stack)
                # Reference: A Learning-based Optimal Market Bidding Strategy for Price-Maker Energy Storage
                if bid.quantity <0 :
                    bid.price=0
                accepted.append(bid)
                
                if verbose: 
                    print(f'Bid accepted. Dispatched for {bid.quantity}. Cumulative capacity now {cumulative_cap_MW}')

            else:
                bid = bid.copy()
                bid.quantity = capacity_MW - cumulative_cap_MW
                accepted.append(bid)
                if verbose: 
                    print(f'Final bid cleared. Dispatched for {bid.quantity}')
                cumulative_cap_MW += bid.quantity
                satisfied = True
                break

        if not satisfied:
            print(f'Demand not met!\n{cumulative_cap_MW}<{capacity_MW}')
            
        return accepted

    def get_all_bids_dict(self):
        """
        Create a dictionary containing all bids in the BidStack.
        """
        out = {}
        for bid in self.stack:
            duid = bid.duid
            out[duid] = [] if not duid in out else out[duid]
            out[duid].append(bid.to_dict())
        return out

class DispatchOrder():#
    def __init__(self, winning_bids):
        self.winning_bids = winning_bids
    
    def get_generator_dispatch(self):
        dispatch = {}
        for bid in self.winning_bids:
            if not bid.duid in dispatch:
                dispatch[bid.duid] = 0
            dispatch[bid.duid] += bid.quantity
        return dispatch

class Market():
    
    def __init__(self, initial_demand_MW):
        """ 
        Takes a list of participant duids (strings), and the initial MW demand.
        """
        self.bidstack = BidStack()
        self.timestep = 0
        #self.duids = duids
        self.demand_MW = initial_demand_MW
        self.step(initial_demand_MW)
    
    def step(self, demand_MW):
        """Called to step the market forward in time by one. """
        self.timestep += 1
        #self.submitted = { p : False for p in self.duids }
        self.demand_MW = demand_MW
        self.bidstack = BidStack()
    
    def reset(self, demand_MW):
        self.timestep = 0
        self.step(demand_MW)
    
    def add_bid(self, bids: List[Bid]):
        """
        Takes an array of bid objects and adds them to the bidstack.
        """
        for bid in bids:
            self.bidstack.add_price_quantity_bid(bid)
            #self.submitted[bid.duid] = True
        
    def dispatch(self, verbose=False):

        # Perform economic dispatch to get a list of winning bids
        winning_bids = self.bidstack.economic_dispatch(self.demand_MW, verbose=verbose)
        
        # Generate a dispatch order object that stores a queriable result of the dispatch. 
        dispatch_order = DispatchOrder(winning_bids)
        
        # Calculate the market price - price of winning bid
        marginal_price = winning_bids[-1].price
        
        # Get a dict containing each gen and amount dispatched
        dispatch = dispatch_order.get_generator_dispatch()

        state = {
            'dispatch':dispatch,
            'price':marginal_price,
            'demand': self.demand_MW,
            'winning_bids': winning_bids,
            'all_bids':self.bidstack.get_all_bids_dict()
        }
    
        return state

class MeritOrderNEM:
    """
    Class to represent a simplified merit order version of the NEM.

    Loads are represented as -MWs, are implicitly added to market demand 
    by reducing the cumulative capacity MW in the market.

    FCAS markets will have >= 0 price and quantity values only.

    """
    def __init__(
            self, 
            data_manager: DataManager
        ):
        """
        Create a merit order version of the NEM.

        Takes a datamanager object to manage all the data.
        """
        self.data_manager = data_manager
        self.timestep = 0

        self.bid_dict = {}
        demand_dict = self._get_demand_dict()
        self.markets: Dict[str, Market] = {m: Market(demand_dict[m]) for m in MARKETS}
        self.terminated = False
        self.loaded_bid_names = []

    @property
    def curr_period_id(self):
        return self.data_manager.di_periodid[self.timestep]

    def get_current_dispatch_interval(self):
        return self.data_manager.dispatch_intervals[self.timestep]

    def set_current_dispatch_interval(self, di):
        """
        Roundabout helper method to set the dispatch interval directly as a string
        """
        idx = self.data_manager.dispatch_intervals.index(pd.Timestamp(di))
        self.timestep = idx

    def build(self, load_bids: bool = False):
        """
        Build the MeritOrderNEM model in order to be run sequentially
        CALL THIS ONLY ONCE
        """
        if load_bids:
            data_to_bid_interface = DataToBidObjInterface(self.data_manager)
            self.bid_dict = data_to_bid_interface.data_to_bids()

    def load_bids_from_pickle(self, fname, verbose=False):
        """
        Yeah the code is getting pretty messy at this point :(

        Load bid_dict from pickle
        """
        if isinstance(fname, list):
            for fn in fname:
                if not fn in self.loaded_bid_names:
                    if verbose: print(f"Loading {fn} into bid_dict")
                    with open(fn, 'rb') as f:
                        bid_dict = pickle.load(f)

                        # python 3.9 merge dictionaries
                        self.bid_dict = self.bid_dict | bid_dict

                        self.loaded_bid_names.append(fn)
                else:
                    print(f'{fn} has been previously loaded.')
        else:
            with open(fname, 'rb') as f:
                self.bid_dict = pickle.load(f)
            self.loaded_bid_names.append(fname)

    def add_bids(self, bid_dict: dict = None):

        di_curr = self.data_manager.dispatch_intervals[self.timestep]

        historical_bids = self.bid_dict[di_curr]
        for m in MARKETS:
            self.markets[m].add_bid(historical_bids[m])
            if bid_dict is not None:
                self.markets[m].add_bid(bid_dict[m])

    def get_forecast_demand_dict(self) -> Dict[str, float]:
        """
        Get the forecast demand as a dictionary

        Returns
        -------
        Dict[str, float]
            forecast demand as a dictionary
        """
        di_curr = self.data_manager.dispatch_intervals[self.timestep]

        # Create dict here bc lazy
        fcas_req = self.data_manager.fcas_requirements_fc
        demand = self.data_manager.regional_demand_fc

        # absolute shambles of data management here... but it workds
        demand_val = demand.loc[demand.SETTLEMENTDATE == di_curr, 'TOTALDEMAND_FC'].reset_index(drop=True).iloc[0]
        
        fcas_req_cols = [f'{service}_FC' for service in FCAS_SERVICES]
        demand_dict = fcas_req.loc[fcas_req.SETTLEMENTDATE == di_curr, fcas_req_cols].to_dict(orient='records')[0]
        demand_dict['ENERGY_FC'] = demand_val

        return demand_dict

    def _get_demand_dict(self):
        di_curr = self.data_manager.dispatch_intervals[self.timestep]

        # Create dict here bc lazy
        fcas_req = self.data_manager.fcas_requirements
        demand = self.data_manager.regional_demand

        # absolute shambles of data management here... but it workds
        demand_val = demand.loc[demand.SETTLEMENTDATE == di_curr, 'TOTALDEMAND'].reset_index(drop=True).iloc[0]
        
        fcas_req_cols = FCAS_SERVICES
        demand_dict = fcas_req.loc[fcas_req.SETTLEMENTDATE == di_curr, fcas_req_cols].to_dict(orient='records')[0]
        demand_dict['ENERGY'] = demand_val

        return demand_dict

    def step(self):
        
        demand_dict = self._get_demand_dict()

        # ADD DEMAND/FCAS REQUIREMENTS FROM DATA MANAGER
        for market in MARKETS:
            self.markets[market].step(demand_dict[market])
        self.timestep += 1

        if not self.timestep < len(self.data_manager.dispatch_intervals):
            self.terminated = True
       

    def dispatch(self):
        """
        Dispatch the full NEM and all associated markets.
        """
        market_dispatch = {}
        for m in MARKETS:
            result = self.markets[m].dispatch()
            market_dispatch[m] = result
        return market_dispatch

    def reset(self):
        self.timestep = 0

        demand_dict = self._get_demand_dict()
        for m in MARKETS:
            self.markets[m].reset(demand_dict[m])

        self.terminated = False

    def get_nem_state(self):
        """
        Return info on the current state of the NEM.
        Returns a string of (Price, Demand, Marginal Setter) per market
        """
        

            
    