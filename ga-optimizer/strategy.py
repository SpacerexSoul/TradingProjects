import numpy as np


def ma_crossover_strategy(prices, short_window, long_window):
    """
    Simple MA crossover strategy.
    Buy when short MA > long MA, sell when short MA < long MA.
    Returns ROI as percentage.
    """
    if len(prices) < long_window:
        return -100
    
    short_ma = np.convolve(prices, np.ones(short_window)/short_window, mode='valid')
    long_ma = np.convolve(prices, np.ones(long_window)/long_window, mode='valid')
    
    # Align arrays
    diff = len(short_ma) - len(long_ma)
    short_ma = short_ma[diff:]
    aligned_prices = prices[long_window-1:]
    
    if len(short_ma) != len(long_ma):
        return -100
    
    # Generate signals
    position = 0
    cash = 10000
    shares = 0
    
    for i in range(1, len(short_ma)):
        if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
            # Buy signal
            if cash > 0:
                shares = cash / aligned_prices[i]
                cash = 0
        elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
            # Sell signal
            if shares > 0:
                cash = shares * aligned_prices[i]
                shares = 0
    
    # Final value
    final_value = cash + shares * aligned_prices[-1]
    roi = ((final_value - 10000) / 10000) * 100
    
    return roi


def create_fitness_function(prices):
    def fitness(individual):
        short_window, long_window = individual
        return ma_crossover_strategy(prices, short_window, long_window)
    return fitness
