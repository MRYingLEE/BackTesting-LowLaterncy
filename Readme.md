# The backtesting introduction
This is to backtest a trading strategy based on simple moving average (MA) cross signal. Strategy will buy 1 share when short moving average line (such as 5) crosses above long moving average line (such as 25), while sell 1 share when short MA crosses below long MA. Also, we set a stop loss rate (such as 2%). If the last holding lost at the stop loss rate, we will close it immediately.

In order to find an optimized parameters, we will test many combinations of  (short MA periods, long MA periods, stop loss level). 


# Why Low Latency in Python
In serial mode, the code takes about 25 minutes to run on my home PC, which has a 12-core CPU and a GTX 1060 GPU.
The efficiency is too slow to accept, so I have to use low latency method.

The effect is obvious. The running time has been shinked to 4 minutes. The speed is nearly 6X times as serial mode.

# Multithreading vs Multiprocessing
There is a good article to compare multithreading and multiprocessing of Python,
https://medium.com/contentsquare-engineering-blog/multithreading-vs-multiprocessing-in-python-ece023ad55a .

Its conclusions are as the following:
  There can only be one thread running at any given time in a python process.
  Multiprocessing is parallelism. Multithreading is concurrency.
  Multiprocessing is for increasing speed. Multithreading is for hiding latency.
  Multiprocessing is best for computations. Multithreading is best for IO.
  
For backtesting is calculation intensive, so after a multithreading testing, I switched to multiprocessing.  

# Why not in Jupyter Notebook

The code in serial mode is in a Jupyter Notebook. But on Windows, Jupyter Notebook doesn't support multiprocessing.

There is good article (https://medium.com/@grvsinghal/speed-up-your-python-code-using-multiprocessing-on-windows-and-jupyter-or-ipython-2714b49d6fac) to discuss a way to overcome it.

For me, when I switched to multiprocessing mode, I use python code directly to minimize the code transforming. 

So the core code in multiprocessing is nearly the same as that in serial mode.

# Why not GPU

It's a good question. I will try it later.
But due to the backtesting is path depent, it's hard to use parallelism to deal with 1 backtesting internally. At the same time, it's natual to use multiprocessing to deal with many backtestings.
