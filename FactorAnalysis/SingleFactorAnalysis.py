#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:26:24 2019

@author: yangwenli
"""
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Day

unfill = lambda s:s.loc[full_date[(full_date >= s.index[0]) & (full_date <= s.index[-1])]]
ffill = lambda s:s.reindex(full_date[(full_date >= s.index[0]) & (full_date <= s.index[-1])], method = 'ffill')

class SFPortfolio:
    
    def __init__(self, factors, hist_data, group_num = 10, balance_time = '1M', weights = 'EW'):
        '''
        factors:factor value to research
        hist_data:history stock data of the whole market or some subset
        group_num:the number of groups dividedaccording to the factor values
        balance_time: balance frequency
        weights: the weights of single factor portfolio, 'EW' means equal weights
        
        _rebalance_date:the date to rebalance the portfolio
        _group:stocks in every group
        _returns:returns of every group and the long-short portfolio
        '''
        self._factors = factors
        self._balance_time = balance_time
        self._weights = weights
        self._hist_data = hist_data.copy()
        self._group_num = group_num
        
        self._IC_data = None
        self._rebalance_date = self._cal_rebalance_date()
        self._group = None
        self._returns = None
        
    def _cal_rebalance_date(self):
        '''
        Calculate the rebalance date from the trade date according to rebalance
        frequency
        '''
        if isinstance(self._balance_time, str):
            #if the parameter _balance_time is an offset string
            date_series = pd.Series(full_date, index = full_date)
            rebalance_list = date_series.resample(self._balance_time).last().dropna()
            if rebalance_list.iloc[0] != full_date[0]:
                rebalance_list = [full_date[0]] + list(rebalance_list.values)
        elif isinstance(self._balance_time, int):
            #if the parameter_balance_time is an integer,
            #then the portfolio will rebalance every _balance_time trade day
            rebalance_list = list(full_date[::self._balance_time])
            if rebalance_list[-1] != full_date[-1]:
                rebalance_list.append(full_date[-1])
        else:
            raise TypeError("The parameter balance_time must be an offset string or an integer")
        return pd.Series(rebalance_list)
    
    def _cal_portfolio_returns_between_balancing(self):
        '''
        Calculate the cumulitive returns of every stock between two balancing
        '''
        print('_cal_portfolio_returns_between_balancing--1',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        used_columns = [] if self._weights == 'EW' else ['weights']
        gross_returns = self._hist_data[['returns', 'group'] + used_columns]
            
        returns_date = gross_returns.index.get_level_values('date')
        portfolio_returns_between_balancing = [0] * (len(self._rebalance_date) - 1)
        
        for i in range(len(self._rebalance_date) - 1):
            #The start and end of a period between balancing
            start_date, end_date = self._rebalance_date[i], self._rebalance_date[i + 1]
            if i == len(self._rebalance_date) - 2: end_date += Day(1)
            #history data during the period
            returns_between_balancing = gross_returns[(returns_date >= start_date) & (returns_date < end_date)]
            returns_between_balancing = (returns_between_balancing['returns'].fillna(0) + 1).groupby('code', group_keys = False).cumprod()
            portfolio_returns_between_balancing[i] = returns_between_balancing
        
        print('_cal_portfolio_returns_between_balancing--2', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        cum_returns_stocks = pd.concat(portfolio_returns_between_balancing).sort_index()
        cum_returns_stocks.name = 'cum_returns'
        cum_returns_stocks = pd.concat([cum_returns_stocks, gross_returns['group']], axis = 1)
        #Calculate the portfolio value if start from 1
        group_data = cum_returns_stocks[['cum_returns', 'group'] + used_columns].groupby(['date', 'group'])
        if self._weights == 'EW':
            cum_returns = group_data.mean()
        else:
            cum_returns =group_data.apply(lambda df:np.average(df.cum_returns, weights = df.weights)) 
        
        print('_cal_portfolio_returns_between_balancing--3', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        cum_returns = cum_returns.unstack(level = 'group')
        returns_date = cum_returns.index = cum_returns.index.get_level_values('date')
        for i in range(len(self._rebalance_date) - 1):
            #The start and end of a period between balancing
            start_date, end_date = self._rebalance_date[i], self._rebalance_date[i + 1]
            if i == len(self._rebalance_date) - 2: end_date += Day(1)
            
            cum_returns_between_balancing = cum_returns[(returns_date >= start_date) & (returns_date < end_date)]
            returns_between_balancing = cum_returns_between_balancing.pct_change()
            if len(cum_returns_between_balancing) != 0:
                returns_between_balancing.iloc[0] = cum_returns_between_balancing.iloc[0] - 1
            
            portfolio_returns_between_balancing[i] = returns_between_balancing
        
        print('_cal_portfolio_returns_between_balancing--4', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        return pd.concat(portfolio_returns_between_balancing).sort_index()
    
    
    def _rank_and_divide(self):
        '''
        Divide the stocks into 10 groups according to the order of the factor
        '''
        print('_rank_and_divide--1', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        factors_rank = self._factors[self._factors.index.get_level_values(level = 'date').isin(self._rebalance_date)]
        factors_rank = factors_rank.groupby('date', group_keys = False).rank(pct = True)
	
        factors_rank = pd.DataFrame(factors_rank).sort_index()
	
        factors_rank.columns = ['group']
        percent_group = 100 // self._group_num
        factors_rank = factors_rank * 100 // percent_group  + 1
        factors_rank[factors_rank == self._group_num + 1] = self._group_num

        factors_rank = factors_rank.reset_index().set_index('date').groupby('code').apply(unfill)
        try:
            del factors_rank['code']
        except KeyError:
            pass

        factors_rank = factors_rank.groupby('code', group_keys = False).fillna(method = 'ffill')
        self._group = pd.DataFrame(factors_rank)
        print('_rank_and_divide--2', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
	
    def _cal_returns(self):
        '''
        Calculate the portfolio returns of every group and the long-short portfolio
        '''
        if self._group is None: self._rank_and_divide()
              
        #下一条语句很慢,必须想办法优化一下
        self._hist_data['group'] = self._group['group']
	
        returns = self._cal_portfolio_returns_between_balancing()
	
        returns.columns = list(range(1, sfp._group_num + 1))
        returns.columns.name = 'group'

        returns['long_short'] = returns[1] - returns[self._group_num]
        self._returns = returns
        
    def sharpe_ratio(self):
        if self._returns is None: self._cal_returns()
        sr = self._returns.mean() / self._returns.std() * (250 ** 0.5)
        return sr
    
    def plot_cum(self, **args):
        if self._returns is None: self._cal_returns()
        fig, ax = plt.subplots()
        cum_returns = (self._returns + 1).cumprod()
        cum_returns[cum_returns.columns[:-1]].plot(ax = ax)
        cum_returns['long_short'].plot(ax = ax, linewidth = 3, color = 'blueviolet', alpha = 0.7)
        return ax
    
    def _mdd(self, s):
        s = (s + 1).cumprod()
        max_value = 1
        max_draw_down = 0
        for value in s:
            draw_down = value / max_value - 1
            if draw_down < max_draw_down: max_draw_down = draw_down
            if value > max_value: max_value = value
        return max_draw_down
    
    def _IC(self):
        IC_data = pd.concat([self._hist_data['returns'], self._factors], axis = 1)
        IC_date_group = IC_data.groupby('date')
        Rank_IC_series = IC_date_group.corr(method = 'spearman').iloc[::2, 1]
        IC_series = IC_date_group.corr().iloc[::2, 1]
        IC_data_frame = pd.concat([Rank_IC_series, IC_series], axis = 1)
        IC_data_frame.columns = ['Rank_IC', 'IC']
        IC_data_frame.index = IC_data_frame.index.get_level_values('date')
        IC_data_frame = IC_data_frame.dropna()
        self._IC_data = IC_data_frame
    
    def IC(self):
        if self._IC_data is None:
            self._IC()
        return self._IC_data
    
    def max_draw_down(self):
        return self._returns.fillna(0).apply(self._mdd)
