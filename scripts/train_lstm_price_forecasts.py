"""
Script to train all LSTM price forecasts
"""

import datetime
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from calliope.prices import construct_prices_from_merit_order_csv
from torch.utils.tensorboard import SummaryWriter
import argparse

from calliope.forecast.lstm import NEMPriceLSTM, create_dataset

from calliope.defaults import MARKETS

prices = construct_prices_from_merit_order_csv(
    path='E:/Code/Calliope/notebooks/merit_order_prices.csv', 
    region_id= 'NSW1', 
    start_date = '2019/01/01', 
    end_date='2019/09/01', 
    create_agc=True
)
price_df = prices.to_dataframe()


def update_stat_leq(curr, best):
    to_return = best
    if curr < best:
        to_return = curr
    return to_return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path',default='scripts/lstm_models',type=str,help='path to save models')
    parser.add_argument('--n_epochs',default=2000,type=int,help='training epoch numnber')
    parser.add_argument('--batch_size',default=512,type=int,help='batch size')
    parser.add_argument('--lookback',default=128,type=float,help='LSTM lookback')
    parser.add_argument('--lookahead',default=1,type=float,help='LSTM lookahead')
    parser.add_argument('--hidden_size',default=64,type=float,help='LSTM hidden_size')
    parser.add_argument('--lr',default=1e-3,type=float,help='optimiser learning rate')
    parser.add_argument('--num_layers',default=2,type=float,help='LSTM num_layers')
    parser.add_argument('--early_stop',default=5,type=int,help='LSTM early stopping')

    args = parser.parse_args()

    # args to pass
    lookback = args.lookback
    lookahead = args.lookahead
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    path = args.path
    early_stop = args.early_stop
    lr = args.lr


    # terrible code to split by time
    split_date = pd.to_datetime('2019-07-01 00:00:00')
    # split_date_val = pd.to_datetime('2019-08-01 00:00:00')

    train_size = prices.idx[pd.to_datetime(prices.SETTLEMENTDATE) == split_date][0]
    # val_idx = prices.idx[pd.to_datetime(prices.SETTLEMENTDATE) == split_date_val][0]

    for m in MARKETS:
        print(f'Training for {m}')

        col = f'{m}ROP'
        if m == 'ENERGY':
            col = 'ROP'

        train, test = price_df.loc[:train_size, col],  price_df.loc[train_size:, col]

        X_train, y_train = create_dataset(train, lookback=lookback, lookahead=lookahead)
        X_test, y_test = create_dataset(test, lookback=lookback, lookahead=lookahead)
        
        # create the NEM price lstm model
        model = NEMPriceLSTM(
            input_size=lookback, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=1

        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), 
            shuffle=True,
            batch_size=batch_size
        )

        # get training date
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(f'{path}/runs/{col}_price_forecast_{timestamp}')

        # track best stats
        best_train_rmse = np.inf
        best_test_rmse = np.inf
        best_train_loss = np.inf
        best_epoch = -1

        for epoch in range(n_epochs):
            running_loss=0
            last_loss=0

            model.train()
            for i, data in enumerate(loader):
                X_batch, y_batch = data
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                running_loss += loss.item()  
                if i %10 == 0:

                    last_loss = running_loss / 1000 # loss per batch
                    #print('  batch {} loss: {}'.format(i + 1, last_loss))
                    tb_x = epoch * len(loader) + i + 1
                    writer.add_scalar('Loss/train', last_loss)
                    running_loss = 0.

            # Validation after 50 epochs
            if epoch % 50 != 0:
                continue

            model.eval()
            with torch.no_grad():
                tb_x = epoch * len(loader)
                y_pred = model(X_train)
                train_rmse = np.sqrt(loss_fn(y_pred, y_train))
                y_pred = model(X_test) 
                test_rmse = np.sqrt(loss_fn(y_pred, y_test))

                test_loss = loss_fn(y_pred, y_test)
                writer.add_scalar('RMSE/train_rmse', train_rmse, tb_x)
                writer.add_scalar('RMSE/test_rmse', test_rmse, tb_x)
                writer.add_scalar('Loss/test', test_loss, tb_x)

            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), f'{path}/lstm_{col}_forecast_{timestamp}_best.pth')

            elif epoch - best_epoch > 50*early_stop:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop
            
            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

            # save the model
            torch.save(model.state_dict(), f'{path}/lstm_{col}_forecast_{timestamp}_{epoch}.pth')

