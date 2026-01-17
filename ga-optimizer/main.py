import yfinance as yf
from datetime import datetime, timedelta

from genetic_algorithm import run_ga
from strategy import create_fitness_function
from visualization import plot_evolution


def main():
    print("\nGA Strategy Optimizer\n")
    
    print("Fetching AAPL data...")
    end = datetime.now()
    start = end - timedelta(days=2*365)
    df = yf.Ticker("AAPL").history(start=start, end=end)
    prices = df['Close'].values
    print(f"Loaded {len(prices)} price points")
    
    fitness_func = create_fitness_function(prices)
    
    print("\nRunning Genetic Algorithm...\n")
    best_params, all_generations = run_ga(fitness_func, pop_size=25, generations=12)
    
    final_roi = fitness_func(best_params)
    print(f"\nBest parameters: Short MA = {best_params[0]}, Long MA = {best_params[1]}")
    print(f"Best ROI: {final_roi:.2f}%")
    
    plot_evolution(all_generations)
    print("\nDone.")


if __name__ == "__main__":
    main()
