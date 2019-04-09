
#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import datetime as dt
from datetime import timedelta as td
#%% [markdown]
# # The workflow: Indicator, Signal and strategy
#%% [markdown]
# ## The column name protocols between different stages and staff
# So that the data source can be interpreted by strategy

#%%
def ma_col_name(nperiods):
    return "MA"+str(nperiods)

def pre_ma_col_name(nperiods):
    return "Pre_"+ma_col_name(nperiods)

def crossup_col_name(ma_short, ma_long):
    return str(ma_short)+"_CU_"+str(ma_long)

def crossdown_col_name(ma_short, ma_long):
    return str(ma_short)+"_CD_"+str(ma_long)

#%% [markdown]
# ## The MA parameters we will use

#%%
ma_range_short=list(range(5,10+1))
ma_range_long=list(range(25,35+1))
ma_range=ma_range_short+ma_range_long
ma_range

#%% [markdown]
# ## The Stoploss level we will use

#%%
stoploss_range=[x/100.0 for x in range(1,5+1)]
stoploss_range=[0.005]+stoploss_range+[1]
stoploss_range

#%% [markdown]
# # Backtesting
#%% [markdown]
# # To Load Market Data Indicators and Signals
# So that the step 1, 2 are independent to back testing

#%%
if __name__ == '__main__':
    df=pd.read_csv("signals.csv" ,parse_dates=["datetime"],index_col="datetime")
    df.sort_index(ascending=True,inplace=True)
# Test df.iloc[0]

#%% [markdown]
# ## Backtesting

#%%
def strategy_test(df, ma_short=5, ma_long=25, stoploss_rate=0.05):
    """Backtest for a set of (MA short periods, MA long periods, stop loss rate)"""
    ticker_holdings = []  #Holding of ticks
    ticker_trades=[]
    ticker_stoploss=[]
    trades_pl=[]
    #stoplosses_pl=[]
    cash_flows=[]  #Cash flow (In when sell and out when buy)
    
    current_holding=0
    previous_cost_Buys=0
    previous_cost_Sells=0
    initial_cash=0  # The first investment as cash 
    crossdown=crossdown_col_name(ma_short,ma_long)
    crossup=crossup_col_name(ma_short,ma_long)
    
    for index, row in df.iterrows():  
        net_cash_flow=0
        trades=0 #non-cover
        stoploss_trades=0 #cover
        trade_pl=0
        #stoploss_pl=0
        
        if stoploss_rate<1: # When stoploss>=1, which means no stoploss!
            # stop loss first
            if current_holding>0 and row["close"]<((1-stoploss_rate)*previous_cost_Buys): # stop loss for long
                stoploss_trades=-current_holding
                current_holding=0
                net_cash_flow=(1-stoploss_rate)*previous_cost_Buys*abs(stoploss_trades)
                #stoploss_pl=-stoploss_rate # The loss
                trade_pl=-stoploss_rate # The loss
            elif current_holding<0 and row['close']>(1+stoploss_rate)*previous_cost_Sells: # stop loss for short
                stoploss_trades=-current_holding
                current_holding=0
                #stoploss_pl=-stoploss_rate
                trade_pl=-stoploss_rate # The loss
                net_cash_flow=-(1+stoploss_rate)*previous_cost_Sells*abs(stoploss_trades)
        
        bool_crossdown=row[crossdown]
        bool_crossup=row[crossup]
        
        # Then we deal with buy and sell signal
        if bool_crossup:
            if (current_holding<0): # cover short
                trade_pl=-(row["close"]/previous_cost_Sells-1)
            
            current_holding=current_holding+1
            net_cash_flow=net_cash_flow-row["close"]
            previous_cost_Buys=row["close"]
            trades=1
            
            
            if initial_cash==0:
                initial_cash=row["close"]
                
        elif bool_crossdown:
            if (current_holding>0): # cover long
                trade_pl=row["close"]/previous_cost_Buys-1
            
            current_holding=current_holding-1
            net_cash_flow=net_cash_flow+row["close"]
            previous_cost_Sells=row["close"]
            trades=-1
            
            if initial_cash==0:
                initial_cash=row["close"]
                  
        ticker_holdings.append(current_holding)
        cash_flows.append(net_cash_flow)
        ticker_trades.append(trades)
        ticker_stoploss.append(stoploss_trades)
        trades_pl.append(trade_pl)
        #stoplosses_pl.append(stoploss_pl)
        
    df_strategy=pd.DataFrame({"close":df["close"],"holding": ticker_holdings,"cash_flows":cash_flows,                           "trades":ticker_trades,"stoploss":ticker_stoploss, "trades_pl":trades_pl                              #,"stoplosses_pl":stoplosses_pl
                             },index=df.index)
                
    df_strategy["cum_cash_flow"]=df_strategy["cash_flows"].cumsum()
    df_strategy["cash"]=pd.Series([initial_cash+cum_cf for cum_cf in df_strategy["cum_cash_flow"]],index=df.index)
    df_strategy["strategy"]=df_strategy.holding*df_strategy.close+df_strategy["cash"]
    df_strategy["strategy_pl_cum"]=df_strategy["strategy"]/initial_cash-1
    
    return df_strategy

#%% [markdown]
# ## We define the metrics 

#%%

# global parameters we share between 
annual_trading_days=252

risk_free=0.01 # for 2018 the low interest rate
risk_free_daily=(1+risk_free)**(1/annual_trading_days)-1 
annualized_parameter=annual_trading_days**0.5

target_daily_return=0 # We use 0 as the target daily return

#%% [markdown]
# ### Sharpe Ratio

#%%
def sharpe_ratio(values):
    """To calculate Sharpe Ratio by portfolio values"""
    daily_return=values.pct_change()
    sharpe_ratio=(daily_return.mean()-risk_free_daily)/daily_return.std()
    sharpe_ratio_annual=annualized_parameter*sharpe_ratio
    return sharpe_ratio,sharpe_ratio_annual

#%% [markdown]
# ### Sortino Ratio

#%%
def sortino_ratio(values,target_return=0):
    """To calculate Sortino Ratio by portfolio values"""    
    daily_return=values.pct_change()
    expected_return = daily_return.mean()

    df_temp=daily_return.to_frame("daily")
    sq_mean=df_temp.applymap(lambda x: (x-target_return)**2 if x<target_return else 0).mean()[0] 
    
    #print(sq_mean)
    down_stdev = np.sqrt(sq_mean)
    
    sortino_ratio = (expected_return - risk_free_daily)/down_stdev
    
    sortino_ratio_annual=annualized_parameter*sortino_ratio
    
    return sortino_ratio, sortino_ratio_annual

#%% [markdown]
# ### Maximum Drawdown

#%%
def max_drawdown(values):
    """To calculate maximum drawdown"""
    max_dd=0
    for i in range(0,len(values)-1):
        min_value=values[i+1:].min()
        drawdown=min_value/values[i]-1
        
        if drawdown<max_dd:
            max_dd=drawdown
    
    if max_dd==0:
        return 0
    else:
        return abs(max_dd) # finally we use positive


#%%
def strategy_metrics(value_serie):
    """To calculate all required performance indicators"""
    sharpe,sharpe_annual=sharpe_ratio(value_serie)
    sortino, sortino_annual=sortino_ratio(value_serie)
    max_dd=max_drawdown(value_serie)
    return sharpe, sortino,max_dd


#%%
if __name__ == '__main__':
    # clear global variables
    df_metrics=pd.DataFrame()
    series=[]
    dfs={}


#%%
def adjust_datatype(df_metrics):
    df_metrics.set_index('strategy_name', inplace=True)
    df_metrics.ma_short=df_metrics.ma_short.astype("int64")
    df_metrics.ma_long=df_metrics.ma_long.astype("int64")
    df_metrics.stoploss=df_metrics.stoploss.astype("float64")
    df_metrics.sharpe_ratio =df_metrics.sharpe_ratio.astype("float64")
    df_metrics.sortino_ratio =df_metrics.sortino_ratio.astype("float64")
    df_metrics.max_drawdown=df_metrics.max_drawdown.astype("float64")
    df_metrics.Final_Return=df_metrics.Final_Return.astype("float64")
    df_metrics.max_drawdown=df_metrics.max_drawdown.astype("float64")
    df_metrics.Total_Trades=df_metrics.Total_Trades.astype("int64")
    df_metrics.Average_Trades_PL=df_metrics.Average_Trades_PL.astype("float64")
    
def all_strategy_test(ma_range_short,ma_range_long,stoploss_range):
    """Backtest parameter combinations of MA short range, MA long range and stop loss range."""
    global df_metrics,series,dfs
    df_metrics=pd.DataFrame()
    series=[]
    dfs={}
    
    for ma_short in ma_range_short:
        for ma_long in ma_range_long:
            for stoploss in stoploss_range:
                strategy_name=str(ma_short)+"_"+str(ma_long)+"_"+str(stoploss)
                df_strategy=strategy_test(df, ma_short, ma_long, stoploss)
                dfs[strategy_name]=df_strategy
                
                sharpe_ratio, sortino,max_dd=strategy_metrics(df_strategy.strategy)
            
                trades_total=(abs(df_strategy.trades)+abs(df_strategy.stoploss)).sum()
                trades_mean=df_strategy.trades_pl.mean()
                final_pl=df_strategy.strategy_pl_cum.iloc[-1]
                    
                series.append(pd.Series({"strategy_name":strategy_name,"ma_short":ma_short,"ma_long":ma_long,\
                                                        "stoploss": stoploss,"sharpe_ratio":sharpe_ratio,"sortino_ratio":sortino,\
                                                        "max_drawdown":max_dd,\
                                        "Final_Return":final_pl,"Total_Trades":trades_total,"Average_Trades_PL":trades_mean\
                                        }))
    
    df_metrics=pd.concat(series,ignore_index=True,axis=1).T                                    
    adjust_datatype(df_metrics)


#%% [markdown]
# ## MultiProcessing (about 3? minutes) Version--To Back Test All Strategy Combinations



#%%
def strategy_test_process(df_dict,series,global_df, ma_short=5, ma_long=25, stoploss_rate=0.05):

    strategy_name=str(ma_short)+"_"+str(ma_long)+"_"+str(stoploss_rate)
    df_strategy=strategy_test(global_df,ma_short,ma_long,stoploss_rate)
    df_dict[strategy_name]=df_strategy
    #df_dict[strategy_name]=strategy_name
    
    sharpe_ratio, sortino,max_dd=strategy_metrics(df_strategy.strategy)
            
    trades_total=(abs(df_strategy.trades)+abs(df_strategy.stoploss)).sum()
    trades_mean=df_strategy.trades_pl.mean()
    final_pl=df_strategy.strategy_pl_cum.iloc[-1]
                    
    series.append(pd.Series({"strategy_name":strategy_name,"ma_short":ma_short,"ma_long":ma_long,\
                                                        "stoploss": stoploss_rate,"sharpe_ratio":sharpe_ratio,"sortino_ratio":sortino,\
                                                        "max_drawdown":max_dd,\
                                        "Final_Return":final_pl,"Total_Trades":trades_total,"Average_Trades_PL":trades_mean\
                                        }))      
    #print(strategy_name+" Average_Trades_PL:"+str(trades_mean))
         

# Step 1: Redefine, to accept `i`, the iteration number
def howmany_within_range2(i, row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return (i, count)

import multiprocessing as mp 
from multiprocessing import Manager

#print(mp.cpu_count())

#print([(ma_short, ma_long, stoploss_rate) for ma_short in list([5,6]) for ma_long in list([25,26]) for stoploss_rate in list([0.01,0.02])])

if __name__ == '__main__':
    mp.freeze_support()
    #mp.set_start_method("spawn")

    pool = mp.Pool(mp.cpu_count())

    processingtest=False
    
    if processingtest:
        # Parallelizing with Pool.starmap_async()
        import numpy as np
        from time import time

        # Prepare data
        #np.random.RandomState(100)
        arr = np.random.randint(0, 10, size=[2000, 5])
        data = arr.tolist()

        #results = []

        results = pool.starmap_async(howmany_within_range2, [(i, row, 4, 8) for i, row in enumerate(data)]).get()
    
        pool.close()
        print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]
    else:
        with Manager() as manager:
            t_begin=dt.datetime.now()
            df_dict = manager.dict()
            series = manager.list()

            #global df_metrics,series,dfs
            #df_metrics=pd.DataFrame()
           
            tasks=[]
            rows=[(df_dict, series, df, ma_short, ma_long, stoploss_rate) \
                for ma_short in ma_range_short for ma_long in ma_range_long for stoploss_rate in stoploss_range]
                #for ma_short in list([5,6]) for ma_long in list([25,26]) for stoploss_rate in list([0.01,0.02])]
            results=[]
            results=pool.starmap_async(strategy_test_process,rows).get()
            t_end=dt.datetime.now()
            print(t_end-t_begin)

            #pool.join()
            pool.close()

            #adjust_datatype(series)
            #df_test=df_dict["10_35_0.04"]
            #print(df_test.head()) 

            df_metrics=pd.concat(series,ignore_index=True,axis=1).T                                    
            adjust_datatype(df_metrics)
            df_metrics.to_csv("metrics-processing.csv")
            #print(df_metrics)