from calliope.optimisation.base import AbstractOptimisationModel
from calliope.market import MeritOrderNEM, Bid
from calliope.prices import Prices, get_prices_over_horizon, construct_prices_from_dict
from calliope.nempy_utils import get_dispatch_intervals
import pandas as pd
from tqdm import tqdm
from calliope.defaults import MARKETS, MARKETS_REG
from typing import List

def calculate_simulation_cumulative_revenue(
        results: pd.DataFrame,
        markets_list: List[str] = MARKETS
    ) -> pd.DataFrame:
    """
    Calculate the cumulative revenue from the simulation.

    Must have columns _dispatch and _price

    Parameters
    ----------
    results : pd.DataFrame
        results from the simulation

    Returns
    -------
    pd.DataFrame
        dataframe of cumulative revenues
    """
    # calculate revenues 
    cumulative_revenue = pd.DataFrame()
    for m in markets_list:
        cumulative_revenue[f'{m}_revenue'] = results[f'{m}_price'] * results[f'{m}_dispatch'] 
    
    cumulative_revenue = cumulative_revenue.cumsum(axis=0)

    cumulative_revenue['TOTAL'] = cumulative_revenue.sum(axis=1)

    # add the settlementdate column
    cumulative_revenue['SETTLEMENTDATE'] = results['SETTLEMENTDATE']
    return cumulative_revenue

def run_bess_mpc_simulation(
        model: AbstractOptimisationModel, 
        prices: Prices, 
        nem: MeritOrderNEM,
        horizon: int =288,
        storage_override: float = None,
        price_forecasts: dict = None


    ) -> pd.DataFrame:
    """
    Run a model predictive control battery model over the Merit Order NEM
    with prices in a rolling horizon.

    Here the model optimises over a finite horizon, and takes the first timestep as the control action to be bid into the NEM.
    """
    nem.reset()
    nem.step()
    bess_name = model.config.station_name
    dispatch_results = pd.DataFrame()
    model.init_storage_override = None
    model.cumulative_override = None
    if storage_override is None:
        storage_track = model.config.init_storage*model.config.capacity *model.config.duration
    else:
        storage_track = storage_override
        model.init_storage_override = storage_override
        
    region_id = prices.REGIONID[0]
    cumulative_track = 0

    previous_results=None

    #idx_rebase =  - prices.idx[0]

    # rebase to 0 as idx is misleading ffs
    for idx in tqdm(prices.idx):
        if idx > len(prices.idx) - horizon:
            print('Not enough horizon periods left. Ending loop.')
            break

        if price_forecasts is None: 
            prices_horizon = get_prices_over_horizon(prices, idx, idx+horizon)
            prices_horizon.idx = prices_horizon.idx - idx
        else:
            # get from the price forecast dataset (has been pre-processed for convenience)
            # NOTE: a more accurate run here is to predict then optimise as market prices are impacted.
            # however due to computational limitations here this method has to be simplified.
            prices_horizon_dict = price_forecasts[pd.Timestamp(prices.SETTLEMENTDATE[idx])]
            prices_horizon = construct_prices_from_dict(prices_horizon_dict, region_id=region_id)
            

        # loop through all rolling horizons
        model.set_parameters(prices_horizon)
        model.build()
        model.solve(verbose=False)

        results = model.to_dataframe()
        # print(results)
        try:
            results_mpc = results[
                ['CHARGE','DISCHARGE','CHARGE_FROM_LOWERREG', 'DISCHARGE_FROM_RAISEREG',
                'LOWERDELAY', 'LOWERFAST', 'LOWERREGULATION', 'LOWERSLOW', 
                'RAISEDELAY', 'RAISEFAST', 'RAISEREGULATION', 'RAISESLOW', 'SOC'
                ]
            ].reset_index(drop=True).iloc[0]
        except:
            return previous_results
        previous_results = results

        # get bid files
        energy = ((results_mpc['DISCHARGE']) - (results_mpc['CHARGE']))
        RAISEFAST = results_mpc.RAISEFAST
        RAISESLOW = results_mpc.RAISESLOW
        RAISEDELAY = results_mpc.RAISEDELAY
        RAISEREGULATION = results_mpc.RAISEREGULATION
        LOWERFAST = results_mpc.LOWERFAST
        LOWERSLOW = results_mpc.LOWERSLOW
        LOWERDELAY = results_mpc.LOWERDELAY
        LOWERREGULATION = results_mpc.LOWERREGULATION
        SOC = results_mpc.SOC

        bess_bid_dict = {
            'ENERGY': [Bid(bess_name, -1000, energy, 1)],
            'RAISE6SEC': [Bid(bess_name, 0.0, RAISEFAST, 1)],
            'RAISE60SEC': [Bid(bess_name, 0.0, RAISESLOW, 1)],
            'RAISE5MIN': [Bid(bess_name, 0.0, RAISEDELAY, 1)],
            'RAISEREG': [Bid(bess_name, 0.0, RAISEREGULATION, 1)],
            'LOWER6SEC': [Bid(bess_name, 0.0, LOWERFAST, 1)],
            'LOWER60SEC': [Bid(bess_name, 0.0, LOWERSLOW, 1)],
            'LOWER5MIN': [Bid(bess_name, 0.0, LOWERDELAY, 1)],
            'LOWERREG': [Bid(bess_name, 0.0, LOWERREGULATION, 1)],
        }

        nem.add_bids(bess_bid_dict)
        dispatch = nem.dispatch()

        def get_bess_dispatch(service):
            bess_dispatch = 0
            try:
                bess_dispatch = dispatch[service]['dispatch'][bess_name]
            except:
                bess_dispatch = 0
            return bess_dispatch
        
        # results
        dispatch_results_di = {}
        for k in bess_bid_dict.keys():
            dispatch_results_di[f'{k}_dispatch'] = get_bess_dispatch(k)
            dispatch_results_di[f'{k}_price'] = dispatch[k]['price']
        dispatch_results_di['SETTLEMENTDATE']=prices.SETTLEMENTDATE[idx]

        energy_dispatch = dispatch_results_di['ENERGY_dispatch']
        rreg_dispatch = dispatch_results_di['RAISEREG_dispatch']
        lreg_dispatch = dispatch_results_di['LOWERREG_dispatch']

        rreg_discharge = rreg_dispatch * model.config.rreg_util
        lreg_charge = lreg_dispatch * model.config.lreg_util

        charge, discharge=0,0
        if energy_dispatch>0:
            discharge = energy_dispatch
        if energy_dispatch <0:
            charge = abs(energy_dispatch)

        eta_c, eta_d = model.config.charge_efficiency, model.config.discharge_efficiency
        intervals = prices.intervals_per_hour

        storage_track = storage_track + (1/intervals)*(((charge+lreg_charge)*eta_c) - ((discharge+rreg_discharge)/eta_d))

        cumulative_track = cumulative_track + discharge/eta_d

        dispatch_results_di['SOC'] = storage_track

        # min_storage = model.config.capacity *model.config.duration*model.config.min_energy
        # max_storage = model.config.capacity *model.config.duration*model.config.max_energy
        # storage_track = max(min_storage, min(storage_track, max_storage))

        dispatch_results_di_df = pd.DataFrame(dispatch_results_di, index=[0])
        dispatch_results = pd.concat([dispatch_results, dispatch_results_di_df]).reset_index(drop=True)

        model.init_storage_override = storage_track
        model.cumulative_override = cumulative_track
        
        nem.step()

        # free memory
        del prices_horizon

    # reset the nem before finishing
    nem.reset()

    return dispatch_results
    


     
    


    

    







