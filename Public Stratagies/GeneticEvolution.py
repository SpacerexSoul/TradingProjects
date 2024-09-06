import random
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Define the trading strategy class
class BalancedTradingStrategy:
    def __init__(self, params: dict):
        self.params = params

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(index=data.index, dtype=int)

        # Calculate short and long EMAs
        short_ema = data['Close'].ewm(span=int(self.params['short_window'])).mean()
        long_ema = data['Close'].ewm(span=int(self.params['long_window'])).mean()

        # MACD and signal line calculation
        macd = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        signal_line = macd.ewm(span=9).mean()

        # RSI calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(com=13).mean()
        loss = -delta.where(delta < 0, 0).ewm(com=13).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals based on EMA, MACD, and RSI logic
        signals[(short_ema > long_ema) & (macd > signal_line) & (rsi < self.params['rsi_oversold'])] = 1
        signals[(short_ema < long_ema) & (macd < signal_line) & (rsi > self.params['rsi_overbought'])] = -1

        # Clip the signals to ensure only -1, 0, 1 values
        signals = signals.clip(lower=-1, upper=1)

        return signals

    def backtest(self, data: pd.DataFrame) -> float:
        signals = self.generate_signals(data)
        if signals.sum() == 0:  # If no signals are generated
            return 0.0  # Return 0 if no trades are made

        # Shift positions by 1 day so trades happen the day after the signal
        positions = signals.shift(1).fillna(0)

        # Calculate daily returns
        returns = data['Close'].pct_change()

        # Apply strategy returns using positions and leverage
        strategy_returns = positions * returns * self.params['leverage']
        cumulative_returns = (1 + strategy_returns).cumprod()

        # Check for invalid or extremely negative returns (blowing the account)
        if cumulative_returns.min() < 0.1:  # Threshold to consider the account blown (e.g., losing 90%)
            return 0.0  # Return 0 if the account is effectively blown

        total_return = cumulative_returns.iloc[-1] - 1
        return max(total_return, 0.0)  # Ensure ROI never goes negative

# Genetic Algorithm for parameter optimization
class GeneticAlgorithm:
    def __init__(self, param_ranges, population_size=100, mutation_rate=0.01, crossover_rate=0.7):
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def create_individual(self):
        return {key: random.uniform(range[0], range[1]) for key, range in self.param_ranges.items()}

    def tournament_selection(self, population, fitnesses, k=3):
        selected = []
        for _ in range(2):
            candidates = random.sample(list(zip(population, fitnesses)), k)
            winner = max(candidates, key=lambda item: item[1])
            selected.append(winner[0])
        return selected

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child = {}
            for key in parent1.keys():
                child[key] = random.choice([parent1[key], parent2[key]])
            return child
        else:
            return random.choice([parent1, parent2])

    def mutate(self, individual):
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                individual[key] = random.uniform(self.param_ranges[key][0], self.param_ranges[key][1])
        return individual

    def evaluate_fitness(self, individual, data):
        strategy = BalancedTradingStrategy(individual)
        roi = strategy.backtest(data)
        return roi

    def optimize(self, data, generations=100):
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_roi = float('-inf')

        for generation in range(generations):
            fitnesses = [self.evaluate_fitness(ind, data) for ind in population]
            if max(fitnesses) > best_roi:
                best_roi = max(fitnesses)
                best_individual = population[fitnesses.index(best_roi)]

            print(f"Generation {generation + 1}: Best ROI = {best_roi * 100:.2f}%")

            # Elitism: Preserve the best individual
            elite = population[fitnesses.index(max(fitnesses))]

            # Create new population
            new_population = [elite]  # Start with the elite individual
            while len(new_population) < self.population_size:
                parent1, parent2 = self.tournament_selection(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return best_individual, best_roi

def download_data(symbol: str, start_date: datetime, end_date: datetime, interval='1d') -> pd.DataFrame:
    return yf.download(symbol, start=start_date, end=end_date, interval=interval)

def main():
    symbol = 'AAPL'
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    # Download 1-minute interval data for the past 2 days
    data = download_data(symbol, start_date, end_date, interval='1d')
    if data.empty:
        print("No data available for backtesting. Exiting.")
        return

    # Define parameter ranges
    param_ranges = {
        'short_window': (5, 50),  # Adjusting for quicker signals
        'long_window': (50, 200),  # Longer window for trends
        'rsi_oversold': (30, 50),  # Loosened RSI thresholds
        'rsi_overbought': (50, 80),
        'leverage': (3, 5)  # Leverage range
    }

    # Set up and run the genetic algorithm
    ga = GeneticAlgorithm(param_ranges, population_size=1000, mutation_rate=0.05, crossover_rate=0.7)
    best_params, best_roi = ga.optimize(data, generations=100)

    print(f"\nBest Parameters: {best_params}")
    print(f"Final ROI: {best_roi * 100:.2f}%")

if __name__ == "__main__":
    main()
