"""
Model the market clearing of the NEM 
inspired by 10.48550/arXiv.2406.00974
"""
import gurobipy as gpy
import time
from calliope.defaults import MARKETS, RAISE_CONTINGENCY_IDX, RAISE_REG_IDX, LOWER_CONTINGENCY_IDX, LOWER_REG_IDX

MARKET_FLOOR = -1000
MARKET_CAP = 16600
BIGM = 1e6


def MarketClearingModel_AdjustedIndexer(mapping, demand_dict, price_bid_dict, vol_bid_dict, max_avail, warm_start={}):
    """
    This represents the static parts of the MILP model that are just all other market data.

    MILP model to clear market and return clearing prices and unit dispatch.

    This model requires that the volume bids are cumulative sums over all the bands as the binary variables 
    here are simpler to select the single band used.
    """

    # Create a new model
    model: gpy.Model = gpy.Model("nemde_model")

    # Sets and indices
    n_markets = len(MARKETS)
    idx_markets = range(n_markets)
    idx_bands = range(10)

    n_bands = 10

    demand = [demand_dict[m] for m in MARKETS]

    # map markets to duids
    #mapping = dict(volume_bids.groupby('BIDTYPE')['DUID'].unique().reset_index().values)

    # get max number of units
    n_units = max(mapping[k].shape[0] for k in mapping.keys())
    n_units_all = {k: mapping[k].shape[0] for k in MARKETS}

    ### Variables
    market_clearing_price = model.addVars(n_markets, lb=MARKET_FLOOR, ub=MARKET_CAP, vtype=gpy.GRB.CONTINUOUS, name=[f'mcp_{i}' for i in MARKETS])
    
    # variables where number of bids in each market changes
    # index n_bands+1 for overflow
    unit_binary = model.addVars(n_markets, n_units, vtype=gpy.GRB.BINARY, name='UnitEnabled')
    unit_binary_band = model.addVars(n_markets, n_units, n_bands, vtype=gpy.GRB.BINARY, name='UnitBandEnabled')
    bigm_binary = model.addVars(n_markets, n_units, vtype=gpy.GRB.BINARY, name='BigMBinary')
    bid_capacity = model.addVars(n_markets, n_units, lb= 0, ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS, name='BidCapacity')
    bid_price = model.addVars(n_markets, n_units, lb= MARKET_FLOOR, ub=MARKET_CAP, vtype=gpy.GRB.CONTINUOUS, name='BidPrice')

    enabled_capacity = model.addVars(n_markets, n_units, lb= 0, ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS, name='EnabledCapacity')
    # Update model after warm starting
    model.update()

    # warm start variables if applicable
    # this will be the result of a simpler merit order market model
    if len(warm_start) >0:
        t=time.time()
        m=0

        # Optimize model
        if "mcp" in warm_start:
            model.getVarByName(f"mcp_{MARKETS[m]}").Start = warm_start['mcp'][m]
        for unit in range(n_units):
            if "UnitEnabled" in warm_start:
                model.getVarByName(f"UnitEnabled[{m},{unit}]").Start = warm_start['UnitEnabled'][m][unit]
            if "EnabledCapacity" in warm_start:
                model.getVarByName(f"EnabledCapacity[{m},{unit}]").Start = warm_start['EnabledCapacity'][m][unit]
            if "BidCapacity" in warm_start:
                model.getVarByName(f"BidCapacity[{m},{unit}]").Start = warm_start['BidCapacity'][m][unit]
        print(f'Warm start time {time.time()-t}')

    model.addConstrs((
        unit_binary[m, unit] == gpy.quicksum([unit_binary_band[m, unit, band] for band in idx_bands if vol_bid_dict[(m, unit, band)] > 0])
        for unit in range(n_units) 
        for m in idx_markets)
        , name = 'UnitBinary'
    )
    model.addConstrs((
        unit_binary[m, unit] <= 1
        for unit in range(n_units)
        for m in idx_markets)
        ,name='SingleUnitBinary'
    )

    # big M method for finding MCP 
    model.addConstrs((
            market_clearing_price[m] 
            >= unit_binary[m,unit]*bid_price[m, unit] 
            for unit in range(n_units) 
            for m in idx_markets)
        , name='BigM_lb'
    )
    model.addConstrs((
        market_clearing_price[m] 
        <= unit_binary[m,unit]*bid_price[m, unit] + BIGM*(1-bigm_binary[m, unit])
        for unit in range(n_units) 
        for m in idx_markets)
        , name='BigM_ub'
    )

    # model.addConstrs((
    #     gpy.quicksum(bigm_binary[m, unit] for unit in range(n_units)) >= 1
    #     for m in idx_markets)
    #     , name = 'BigM_sum'
    # )

    # sum over all unit bianries to find total bid capacity of a unit
    model.addConstrs((
        bid_capacity[m, unit] == gpy.quicksum([unit_binary_band[m, unit, band] * vol_bid_dict[(m, unit, band)] for band in idx_bands if vol_bid_dict[(m, unit, band)] > 0]) 
        for unit in range(n_units) 
        for m in idx_markets)
        , name='BidCapacityConstr'
    )
    
    # sum over all unit pb binaries to find bid price of a unit
    model.addConstrs((
        bid_price[m, unit] == gpy.quicksum([unit_binary_band[m, unit, band] * price_bid_dict[(m, unit, band)] for band in idx_bands if vol_bid_dict[(m, unit, band)] > 0]) 
        for unit in range(n_units) 
        for m in idx_markets)
        , name='BidPriceConstr'
    )

    # supply must equal demand
    model.addConstrs((
        (gpy.quicksum(enabled_capacity[m, unit] for unit in range(n_units)) - demand[m]) == 0.0
        for m in idx_markets) ,
        name='DemandConstr'
    )

    model.addConstrs((
        enabled_capacity[m, unit] <= bid_capacity[m, unit]
        for unit in range(n_units)
        for m in idx_markets) 
    , name='EnabledQuantity'
    )

    # # Max avail constraint
    model.addConstrs((
        bid_capacity[m, unit] <= max_avail[m, unit] 
        for m in idx_markets
        for unit in range(n_units)
        ) 
    , name='MaxAvailBid'
    )

    # Joint capacity and regulation constraints
    model.addConstrs((
        gpy.quicksum([enabled_capacity[m, unit] for m in RAISE_CONTINGENCY_IDX+[0,RAISE_REG_IDX]]) <= max_avail[0, unit] 
        for unit in range(n_units)
        ) 
    , name='JointCapacityRaise'
    )

    # Joint capacity and regulation constraints
    model.addConstrs((
        gpy.quicksum([enabled_capacity[m, unit] for m in LOWER_CONTINGENCY_IDX+[0,LOWER_REG_IDX]]) <= max_avail[0, unit] 
        for unit in range(n_units)  
        ) 
    , name='JointCapacityLower'
    )

    # Objective is to minimize total system cost across all markets
    objective = gpy.quicksum([market_clearing_price[m]*demand[m] for m in idx_markets])
    model.setObjective(objective, gpy.GRB.MINIMIZE)

    # Parameters
    model.Params.Presolve =1
    model.Params.MIPGap=0.05
    model.Params.MIPFocus=2
    
    t=time.time()
    # Optimize model
    model.optimize()
    print(f'Optimise time {time.time()-t}')


    return model

def MarketClearingModel_RawCorrect(mapping, demand_dict, price_bid_dict, vol_bid_dict, max_avail):
    """
    This represents the static parts of the MILP model that are just all other market data.

    MILP model to clear market and return clearing prices and unit dispatch.
    """

    # Create a new model
    model: gpy.Model = gpy.Model("nemde_model")

    # Sets and indices
    n_bands = 10
    n_markets = len(MARKETS)
    idx_markets = range(n_markets)
    idx_bands = range(n_bands)

    demand = [demand_dict[m] for m in MARKETS]

    # map markets to duids
    #mapping = dict(volume_bids.groupby('BIDTYPE')['DUID'].unique().reset_index().values)

    # get max number of units
    n_units = max(mapping[k].shape[0] for k in mapping.keys())
    n_units_all = {k: mapping[k].shape[0] for k in MARKETS}

    ### Variables
    market_clearing_price = model.addVars(n_markets, lb=MARKET_FLOOR, ub=MARKET_CAP, vtype=gpy.GRB.CONTINUOUS, name=[f'mcp_{i}' for i in MARKETS])
    
    # variables where number of bids in each market changes
    # index n_bands+1 for overflow
    unit_binary = model.addVars(n_markets, n_units, n_bands+1, vtype=gpy.GRB.BINARY, name='UnitEnabled')

    bigm_binary = model.addVars(n_markets, n_units, vtype=gpy.GRB.BINARY, name='BigMBinary')
    unit_pb_binary = model.addVars(n_markets, n_units, n_bands, vtype=gpy.GRB.BINARY, name='UnitPriceBandBinary')


    bid_capacity = model.addVars(n_markets, n_units, lb= -gpy.GRB.INFINITY, ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS, name='BidCapacity')
    bid_price = model.addVars(n_markets, n_units, lb= MARKET_FLOOR, ub=MARKET_CAP, vtype=gpy.GRB.CONTINUOUS, name='BidPrice')

    enabled_capacity = model.addVars(n_markets, n_units, lb= 0, ub=gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS, name='EnabledCapacity')

    model.addConstrs((
        sum(unit_pb_binary[m, unit, band] for band in idx_bands ) <= 1
        for unit in range(n_units)
        for m in idx_markets)
        ,name='SingleUnitBinary'
    )

    # "11"th band binary must be 0
    model.addConstrs((
        unit_binary[m, unit, n_bands] == 0
        for unit in range(n_units) 
        for m in idx_markets)
        , name='UnitBinaryEdge0'
    )

    # track final price bid as difference between sequence of unit binaries
    model.addConstrs((
        unit_pb_binary[m, unit, band] == (unit_binary[m, unit, band] - unit_binary[m, unit, band+1])
        for band in idx_bands
        for unit in range(n_units) 
        for m in idx_markets)
        , name = 'UnitPBBinary'
    )

    # enforce bid ordering (i.e. if PB6 clears, then volume from PB1-6 is summed)
    model.addConstrs((
        unit_binary[m, unit, band] >= unit_binary[m, unit, band+1] 
        for band in idx_bands
        for unit in range(n_units) 
        for m in idx_markets)
        , name = 'UnitBinary'
    )

    # big M method for finding MCP 
    model.addConstrs((
            market_clearing_price[m] 
            >= sum(unit_pb_binary[m, unit, band] for band in idx_bands )*bid_price[m, unit] 
            for unit in range(n_units) 
            for m in idx_markets
            )
        , name='BigM_lb'
    )
    model.addConstrs((
        market_clearing_price[m] 
        <= (sum(unit_pb_binary[m, unit, band] for band in idx_bands )*bid_price[m, unit] + BIGM*(1-bigm_binary[m, unit]))
        for unit in range(n_units) 
        for m in idx_markets)
        , name='BigM_ub'
    )

    # model.addConstrs((
    #     sum(bigm_binary[m, unit] for unit in range(n_units)) >= 1
    #     for m in idx_markets)
    #     , name = 'BigMSum'
    # )

    # sum over all unit bianries to find total bid capacity of a unit
    model.addConstrs((
        bid_capacity[m, unit] == sum(unit_binary[m, unit, band] * vol_bid_dict[(m, unit, band)] for band in idx_bands) 
        for unit in range(n_units) 
        for m in idx_markets)
        , name='BidCapacityConstr'
    )

    # sum over all unit pb binaries to find bid price of a unit
    model.addConstrs((
        bid_price[m, unit] == sum(unit_pb_binary[m, unit, band] * price_bid_dict[(m, unit, band)] for band in idx_bands) 
        for unit in range(n_units) 
        for m in idx_markets)
        , name='BidPriceConstr'
    )

    # supply must equal demand
    model.addConstrs((
        sum(enabled_capacity[m, unit] for unit in range(n_units)) == demand[m] 
        for m in idx_markets) ,
        name='DemandConstr'
    )

    model.addConstrs((
        enabled_capacity[m, unit] <= bid_capacity[m, unit]
        for unit in range(n_units)
        for m in idx_markets) 
    , name='EnabledQuantity'
    )

    # Max avail constraint
    model.addConstrs((
        enabled_capacity[m, unit] <= max_avail[m, unit] 
        for m in idx_markets
        for unit in range(n_units)
        ) 
    , name='MaxAvail'
    )

    # Joint capacity and regulation constraints
    model.addConstrs((
        sum(enabled_capacity[m, unit] for m in RAISE_CONTINGENCY_IDX+[0,RAISE_REG_IDX]) <= max_avail[0, unit] 
        for unit in range(n_units)
        ) 
    , name='JointCapacityRaise'
    )


    # Joint capacity and regulation constraints
    model.addConstrs((
        sum(enabled_capacity[m, unit] for m in LOWER_CONTINGENCY_IDX+[0,LOWER_REG_IDX]) <= max_avail[0, unit] 
        for unit in range(n_units)  
        ) 
    , name='JointCapacityLower'
    )
    
    # Objective is to minimize total system cost across all markets
    objective = sum(market_clearing_price[m]*demand[m] for m in idx_markets)
    model.setObjective(objective, gpy.GRB.MINIMIZE)

    # Parameters
    model.Params.Presolve =2
    model.Params.MIPGap=0.1
    model.Params.MIPFocus=2

    
    t=time.time()
    # Optimize model
    model.optimize()
    print(f'Optimise time {time.time()-t}')


    return model

