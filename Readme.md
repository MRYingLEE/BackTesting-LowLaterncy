# The backtesting introduction
This is to backtest a trading strategy based on simple moving average (MA) cross signal. Strategy will buy 1 share when short moving average line (such as 5) crosses above long moving average line (such as 25), while sell 1 share when short MA crosses below long MA. Also, we set a stop loss rate (such as 2%). If the last holding lost at the stop loss rate, we will close it immediately.

In order to find an optimized parameters, we will test many combinations of  (short MA periods, long MA periods, stop loss level). 
![1 Long Short](https://github.com/MRYingLEE/BackTesting-LowLaterncy/blob/master/1Name.png "1 Long Short")
![All sets](https://github.com/MRYingLEE/BackTesting-LowLaterncy/blob/master/Names.png "All sets")

![Heatmap](https://github.com/MRYingLEE/BackTesting-LowLaterncy/blob/master/heatmap1.png "Heatmap of Sharpe Ratio")
![Trade PL](https://github.com/MRYingLEE/BackTesting-LowLaterncy/blob/master/tradepl.png "Trade PL")

## In practice, can I use the result to start trading? 
 
1. Short may be not allowed by some trading platform. And margin cost has to be considered.
2. There is no trading cost included. Trading costs not only include commissions, fees, and taxes, but also price spread and market impact. In the test, the best set by average trades PL is (6,27, 0.5%), whose average trades PL is only 1.17%%. If the total trading cost is 1%%, nearly all profit will be swallowed.
 
3. Technical analysis believes the history repeats. But the history doesn’t repeat simply. The optimized set of parameters could change from time to time. Even when we change the test period, the results could change too.

4. Also, when people find some certain rules to make profit, the trading will correct the market, especially when more and more people find the rules. The trading opportunity will fade away.

4. We need to find robust strategy rules. The cross down/up strategy is too simple. More indicators, signals should be taken into consideration. Also, complicated signal capture method, such as machine learning, deep learning and reinforcement learning may help.

So I suggest not to use the results into trading. Instead the results can be a base for the further research.

## How to improve by using moving average cross signal? 
By using moving average cross signal, we may try some new ideas:

1.	Regarding indicators

a.	We may use different time series. So far the time series are in time domain with an equal distance. In the book, Advances in financial machine learning / Marcos López de Prado, the author argued to use volume to make time series, in other words, time series with variable distance.

b.	We may use different moving average method, such as Cumulative moving average, Weighted moving average, Exponential moving average. Especially, volume data should play an important role. So VWAP (Volume Weighted Average Price) is a good candidate.

c.	Other technical analysis indicators could be used together too.

d.	Other macro indicators, such as interest rate, jobless rate, gold price, primary stock indices can be used together also.

2.	Regarding signals

a.	We may use volume as an additional signal factor. For example, when cross over, the volume should be bigger than before.

b.	We may use signal combinations.

3.	Regarding strategy

a.	We may use profit taking to protect our profit.

b.	We may put different size of orders for different signal levels.

c.	We may use trend checking to avoid directional mistake.

# Why Low Latency in Python
In serial mode, the code takes about 6 minutes to run on my home PC, which has a 12-core CPU and a GTX 1060 GPU.
The efficiency is too slow to accept, so I have to use low latency method.

The effect is obvious. The running time has been shinked to 2 minutes. The speed is nearly 3X times as serial mode.

## Multithreading vs Multiprocessing
There is a good article to compare multithreading and multiprocessing of Python,
https://medium.com/contentsquare-engineering-blog/multithreading-vs-multiprocessing-in-python-ece023ad55a .

Its conclusions are as the following:

  There can only be one thread running at any given time in a python process.
  
  Multiprocessing is parallelism. Multithreading is concurrency.
  
  Multiprocessing is for increasing speed. Multithreading is for hiding latency.
  
  Multiprocessing is best for computations. Multithreading is best for IO.
  
  
For backtesting is calculation intensive, so after a multithreading testing, I switched to multiprocessing.  

## Why not in Jupyter Notebook for Multithreading

The code in serial mode is in a Jupyter Notebook. But on Windows, Jupyter Notebook doesn't support multiprocessing.

There is good article (https://medium.com/@grvsinghal/speed-up-your-python-code-using-multiprocessing-on-windows-and-jupyter-or-ipython-2714b49d6fac) to discuss a way to overcome it.

For me, when I switched to multiprocessing mode, I use python code directly to minimize the code transforming. 

So the core code in multiprocessing is nearly the same as that in serial mode.

## Why not GPU? Failed
Due to the backtesting is path depent, it's hard to use parallelism to deal with 1 backtesting internally. At the same time, it's natual to use multiprocessing to deal with many backtestings.

Even so, I tried to use GPU programming. I chose cuDF (https://github.com/rapidsai/cudf) as the GPU library for it was claimed as "almost a drop-in, API-compatible, GPU-accelerated replacement for pandas".

1. Due to cuDF only works for Linux, I setup a python environment with GPU support, which took me nearly 1 day due to a few technical reasons and version compatibility.

2. There are a lot of important functions, such as CUMMAX/ CUMSUM, were missed in cuDF. It took me a lot of time to rewrite my existing Pandas code.

3. There are a lot of minor but critical differences between cuDF and Pandas. It took me a lot of time to check whether cuDF code worked as I expected.

4. After hours of struggling, the core code started to run successfully, but the performance was very bad, times slower than my Pandas version. I guess for small dataset, the overhead of cuDF is very high.

So, just as I expected GPU is not good for backtesting. Of course, just as I tried before, GPU does much good to deep learning, which is calculation intensive.

