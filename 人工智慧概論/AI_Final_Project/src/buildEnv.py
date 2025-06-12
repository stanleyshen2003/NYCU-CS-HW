import numpy as np
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
import pandas as pd

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]

    prices = env.df['Close'].to_numpy()
    prices = prices[start:end]
    signal_features = env.df.loc[:, ['Open', 'Close', 'High', 'Low']].to_numpy()[start:end]
    return prices, signal_features

def my_calculate_reward(self, action):
    
    '''
    # this is the original one
    step_reward = 0
    trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
        (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True
    '''
    '''
    for i in range(13,3,-2):
        ismin, ismax = self.knowIs(i)
        if(ismax and action == Actions.Buy.value):
            return (i*-5 + 15)/10
        if(ismin and action == Actions.Sell.value):
            return (-5 * i + 15)/10
    '''
    '''  
    if trade:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price
        #step_reward = 0.5
        #if action == Actions.Sell.value:
        #    step_reward += price_diff
        #if action == Actions.Sell.value and ismax:
        #    step_reward += 3
        #if action == Actions.Buy.value and ismin:
        #    step_reward += 3
        #else:
        #    step_reward -= price_diff
    
    
    
    return step_reward
    '''

    
    '''trade = False
    if ((action == Actions.Buy.value and self._position == Positions.Short) or
        (action == Actions.Sell.value and self._position == Positions.Long)):
        trade = True

    if trade:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price

        if self._position == Positions.Long:
            step_reward += price_diff

    return step_reward
'''
    
    if action == Actions.Sell.value and self._current_tick!= self._end_tick:
        return self.prices[self._current_tick] - self.prices[self._current_tick+1]
    if action == Actions.Buy.value and self._current_tick!= self._end_tick:
        return self.prices[self._current_tick+1] - self.prices[self._current_tick]
    return 0
    

class MyStocksEnv(StocksEnv):
    _process_data = my_process_data
    _calculate_reward = my_calculate_reward
    def knowIs(self, window):
        ismax = False
        ismin = False 
        if self._current_tick < self._end_tick-window/2:
            ismax = True
            ismin = True
            for i in range(int(-window/2),int(window/2+1)):
                if(self.prices[self._current_tick + i] > self.prices[self._current_tick]):
                    ismax = False
                if(self.prices[self._current_tick + i] < self.prices[self._current_tick]):
                    ismin = False
        return ismax, ismin
    
    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit) / last_trade_price
                self._total_profit = (shares) * current_price


def state_preprocess(state):
    state = state.reshape(-1)
    tempstate = state
    for i in range(12):
        for j in range(4):
            tempstate[i*4+j] = (state[44+j] - state[i*4+j])/state[44+j]
    return tempstate

def createEnv(stock_no, window_size = 12, frame_bounds = (12, 1000)):
    csv_name = './dataset/stock_data_' + str(stock_no) + '.csv'
    data = pd.read_csv(csv_name)
    read_df = pd.DataFrame(data)
    read_df = read_df.loc[::-1].reset_index(drop=True)
    env = MyStocksEnv(df = read_df, window_size = window_size, frame_bound = frame_bounds)
    return env
if __name__ == "__main__":
    createEnv(2330)
