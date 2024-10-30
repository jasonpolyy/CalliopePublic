import torch
import numpy as np
import datetime
from calliope.prices import construct_prices_from_dict, get_prices_over_horizon
import pandas as pd
from calliope.defaults import MODELS_PATH, BEST_MODELS
import multiprocessing as mp
from tqdm import tqdm
import pickle

def create_dataset(dataset, lookback, lookahead=1):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+lookback:i+lookback+lookahead]
        X.append(feature)
        y.append(target)
    X_arr, y_arr = np.array(X), np.array(y)
    return torch.tensor(X_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.float32)

class NEMPriceLSTM(torch.nn.Module):
    def __init__(self, input_size=128, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.linear = torch.nn.Linear(hidden_size,output_size)
    def forward(self, X):
        x, _ = self.lstm(X)
        x = self.linear(x)
        return x
    
def predict_prices(best_models, price_df, lookback, lookahead, region_id='NSW1'):
    """
    Predict prices given the LSTM model and a dictionary of the best models.
    """
    pred_dict = {}
    for k, v in best_models.items():

        model = NEMPriceLSTM(input_size=128, hidden_size=64, num_layers=2, output_size=1)
        load_path = f'{MODELS_PATH}{v}'
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)

        X_list=price_df[k][-lookback:].values.tolist()
        predictions = []

        for i in range(lookahead):
            X_full_i = torch.tensor(X_list[-lookback:], dtype=torch.float32).unsqueeze(0)
            pred_i = model(X_full_i)
            predictions.append(pred_i.detach().squeeze().item())
            X_list.append(pred_i)
            
        pred_dict[k] = predictions
        
    start_settlementdate = price_df['SETTLEMENTDATE'].iloc[-1]

    settlement_dates = [start_settlementdate+ datetime.timedelta(minutes=5)]
    for i in range(lookahead-1):
        settlement_dates.append(settlement_dates[i] + datetime.timedelta(minutes=5))

    pred_dict['SETTLEMENTDATE'] = np.array(pd.to_datetime(settlement_dates))

    # price_predictions = construct_prices_from_dict(pred_dict, region_id=region_id)
    return pred_dict

def predict_prices_over_horizon(
        model_dict, 
        prices, 
        lookback, 
        lookahead, 
        region_id='NSW1',
        periodic_save=False,
        fn_dict_save='predictions_all_dict.pkl'):

    print(f'Saving to {fn_dict_save}')
    predictions_all_dict = {}
    for i in tqdm(range(len(prices.ROP) - lookback)):
        prices_horizon = get_prices_over_horizon(prices, i+1, i+lookback+1)
        settlementdate = pd.to_datetime(prices_horizon.SETTLEMENTDATE[lookback-1])
        
        prices_horizon_df = prices_horizon.to_dataframe()

        # loop through evaluation period and create all prices for forecast
        predictions = predict_prices(model_dict, prices_horizon_df, lookback=lookback, lookahead=lookahead, region_id=region_id)

        predictions_all_dict[settlementdate] = predictions

        if periodic_save:
            with open(fn_dict_save, 'wb') as f:
                pickle.dump(predictions_all_dict, f)
    
    return predictions_all_dict

    


# BROKEN DOESNT WORK DONT USE THIS I'VE HAD ENOUGH
def predict_in_parallel(best_models, price_df, lookback, lookahead, region_id='NSW1'):

    sims = []

    for model in best_models:
        model_dict = {model: best_models[model]}
        sim = (model_dict, price_df, lookback, lookahead, region_id)
        sims.append(sim)

    results= []

    with mp.Pool(processes=12) as pool:
        for result in pool.starmap(predict_prices_over_horizon, sims):
            results.append(result)

    return results  
    
